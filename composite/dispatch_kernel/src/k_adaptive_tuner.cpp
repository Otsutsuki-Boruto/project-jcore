#include "adaptive_tuner.h"
#include "jcore_isa_dispatch.h"
#include "cpu_features.h"
#include "ffm_cache_block.h"
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <string>
#include <algorithm>
#include <tuple>
#include <functional>
#include <thread>
#include <cstdlib>
#include <iostream>

// ================================================================
// Tuple hash for cache
// ================================================================
struct TupleHash
{
  std::size_t operator()(const std::tuple<size_t, size_t, size_t> &t) const noexcept
  {
    auto h1 = std::hash<size_t>{}(std::get<0>(t));
    auto h2 = std::hash<size_t>{}(std::get<1>(t));
    auto h3 = std::hash<size_t>{}(std::get<2>(t));
    return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
  }
};

// ================================================================
// Internal Static State
// ================================================================
static std::unordered_map<std::string, jcore_matmul_f32_fn> matmul_registry;
static std::unordered_map<std::string, jcore_features_t> matmul_feats;
static std::unordered_map<std::tuple<size_t, size_t, size_t>, std::string, TupleHash> best_kernel_cache;
static std::mutex at_mutex;
static jcore_features_t host_features_cached = 0;
static bool at_initialized = false;

// Cache info cached at init
static size_t l1_cache_size = 32768;   // 32KB default
static size_t l2_cache_size = 262144;  // 256KB default
static size_t l3_cache_size = 8388608; // 8MB default

extern "C"
{

  at_status_t at_init(void)
  {
    std::lock_guard<std::mutex> lock(at_mutex);
    if (at_initialized)
      return AT_OK;

    jcore_init_dispatch();

    CPUFeatures f = detect_cpu_features();
    host_features_cached = 0;
    if (f.avx)
      host_features_cached |= JCORE_FEAT_AVX;
    if (f.avx2)
      host_features_cached |= JCORE_FEAT_AVX2;
    if (f.avx512)
      host_features_cached |= JCORE_FEAT_AVX512;
    if (f.amx)
      host_features_cached |= JCORE_FEAT_AMX;

    // Cache hierarchy detection (safe API-based)
    ffm_cache_info_t *cache_info = ffm_cache_init();
    if (cache_info)
    {
      // Estimate cache capacities using the API
      size_t elem_size = sizeof(float);

      // Compute approximate usable bytes per cache level
      // based on tile sizes (these are safe proxies)
      size_t l1_tile = ffm_cache_compute_tile(cache_info, 1, elem_size, 0.9);
      size_t l2_tile = ffm_cache_compute_tile(cache_info, 2, elem_size, 0.8);
      size_t l3_tile = ffm_cache_compute_tile(cache_info, 3, elem_size, 0.7);

      // Convert tile dimensions to approximate cache capacity in bytes
      if (l1_tile > 0)
        l1_cache_size = l1_tile * l1_tile * elem_size;
      if (l2_tile > 0)
        l2_cache_size = l2_tile * l2_tile * elem_size;
      if (l3_tile > 0)
        l3_cache_size = l3_tile * l3_tile * elem_size;

      ffm_cache_free(cache_info);
    }
    else
    {
      std::cerr << "[AT_INIT] Cache detection failed, using defaults.\n";
    }

    // Threading setup - use persistent strings to avoid dangling pointers
    unsigned int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0)
      hw_threads = 4; // fallback

    // Store as static to ensure lifetime
    static std::string blis_threads = std::to_string(hw_threads);
    static std::string openblas_threads = std::to_string(hw_threads);
    static std::string omp_threads = std::to_string(hw_threads);

    // BLIS uses OpenMP by default
    setenv("BLIS_NUM_THREADS", blis_threads.c_str(), 1);
    setenv("OMP_NUM_THREADS", omp_threads.c_str(), 1);

    // OpenBLAS threading - prefer pthreads if available
    setenv("OPENBLAS_NUM_THREADS", openblas_threads.c_str(), 1);
    setenv("GOTO_NUM_THREADS", openblas_threads.c_str(), 1); // legacy fallback

    at_initialized = true;

    std::cerr << "[AT_INIT] Hardware threads: " << hw_threads << "\n";
    std::cerr << "[AT_INIT] Host features: AVX=" << !!(host_features_cached & JCORE_FEAT_AVX)
              << " AVX2=" << !!(host_features_cached & JCORE_FEAT_AVX2)
              << " AVX512=" << !!(host_features_cached & JCORE_FEAT_AVX512) << "\n";
    std::cerr << "[AT_INIT] Cache: L1=" << l1_cache_size
              << " L2=" << l2_cache_size
              << " L3=" << l3_cache_size << "\n";

    return AT_OK;
  }

  void at_shutdown(void)
  {
    std::lock_guard<std::mutex> lock(at_mutex);
    matmul_registry.clear();
    matmul_feats.clear();
    best_kernel_cache.clear();
    at_initialized = false;
  }

  at_status_t at_register_matmul_impl(const char *name,
                                      unsigned long long required_features,
                                      jcore_matmul_f32_fn fn)
  {
    if (!name || !fn)
      return AT_ERR_INVALID_ARG;

    std::lock_guard<std::mutex> lock(at_mutex);
    matmul_registry[name] = fn;
    matmul_feats[name] = required_features;

    std::cerr << "[AT_REGISTER] Kernel: " << name
              << " Features: 0x" << std::hex << required_features << std::dec << "\n";

    return AT_OK;
  }

  // Helper: compute optimal tile size for given cache level
  static size_t compute_dynamic_tile(size_t M, size_t N, size_t K,
                                     size_t cache_size, double occupancy)
  {
    size_t max_dim = std::max({M, N, K});
    size_t usable = static_cast<size_t>(cache_size * occupancy);
    size_t tile_elements = usable / (3 * sizeof(float));
    size_t tile = 1;

    while (tile * tile < tile_elements && tile < max_dim / 2)
      tile *= 2;

    tile = std::min(tile, max_dim);
    return (tile < 8) ? 8 : tile;
  }

  // Kernel selection with proper feature matching and cache-aware heuristics
  at_status_t at_benchmark_matmul_all(size_t M, size_t N, size_t K,
                                      size_t preferred_threads,
                                      size_t tile_size_hint,
                                      char *best_name, size_t best_name_len)
  {
    if (!at_initialized)
      return AT_ERR_NOT_INITIALIZED;

    if (matmul_registry.empty() || !best_name)
      return AT_ERR_NO_KERNELS;

    std::lock_guard<std::mutex> lock(at_mutex);

    auto key = std::make_tuple(M, N, K);
    auto found = best_kernel_cache.find(key);
    if (found != best_kernel_cache.end())
    {
      strncpy(best_name, found->second.c_str(), best_name_len - 1);
      best_name[best_name_len - 1] = '\0';
      return AT_OK;
    }

    // ========================================================
    // Cache-aware tile computation
    // ========================================================
    size_t opt_tile_l1 = compute_dynamic_tile(M, N, K, l1_cache_size, 0.8);
    size_t opt_tile_l2 = compute_dynamic_tile(M, N, K, l2_cache_size, 0.7);
    size_t opt_tile_l3 = compute_dynamic_tile(M, N, K, l3_cache_size, 0.5);

    size_t max_dim = std::max({M, N, K});
    size_t min_dim = std::min({M, N, K});
    size_t working_set_mb = (M * K + K * N + M * N) * sizeof(float) / (1024 * 1024);

    std::cerr << "\n[AT_SELECT] Shape=[" << M << "x" << N << "x" << K
              << "] Working_set=" << working_set_mb << "MB"
              << " Tiles: L1=" << opt_tile_l1
              << " L2=" << opt_tile_l2
              << " L3=" << opt_tile_l3 << "\n";

    // ========================================================
    // Kernel selection strategy based on problem characteristics
    // ========================================================

    struct KernelScore
    {
      std::string name;
      int score;
      bool feature_compatible;
    };

    std::vector<KernelScore> candidates;

    for (const auto &kv : matmul_registry)
    {
      const std::string &name = kv.first;
      jcore_features_t req = matmul_feats[name];

      // Feature compatibility check
      bool compatible = (req & host_features_cached) == req;
      if (!compatible)
        continue;

      int score = 0;

      // Scoring heuristics based on kernel characteristics
      if (name == "blis_sgemm")
      {
        // BLIS: Good general purpose, excellent OpenMP scaling
        if (max_dim <= 256)
          score = 70;
        else if (max_dim <= 2048)
          score = 90;
        else
          score = 85; // still good for large

        // Bonus for multi-threaded workloads
        if (preferred_threads > 1 && max_dim >= 512)
          score += 15;

        // Bonus for L2-cache-friendly sizes
        if (max_dim >= opt_tile_l2 && max_dim <= opt_tile_l2 * 8)
          score += 10;
      }

      else if (name == "openblas_sgemm")
      {
        // OpenBLAS: Best for large matrices, mature threading
        if (max_dim <= 256)
          score = 60;
        else if (max_dim <= 1024)
          score = 80;
        else if (max_dim <= 4096)
          score = 95;
        else
          score = 100; // excellent for very large

        // Strong bonus for large multi-threaded workloads
        if (preferred_threads > 1 && max_dim >= 1024)
          score += 20;

        // Bonus for L3-cache and beyond
        if (working_set_mb >= (l3_cache_size / (1024 * 1024)))
          score += 15;
      }

      candidates.push_back({name, score, compatible});
    }

    if (candidates.empty())
      return AT_ERR_NO_KERNELS;

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const KernelScore &a, const KernelScore &b)
              {
                return a.score > b.score;
              });

    std::string selected = candidates[0].name;

    std::cerr << "[AT_SELECT] Candidates:\n";
    for (const auto &c : candidates)
      std::cerr << "  " << c.name << ": score=" << c.score << "\n";
    std::cerr << "[AT_SELECT] Selected: " << selected << " (score=" << candidates[0].score << ")\n";

    best_kernel_cache[key] = selected;
    strncpy(best_name, selected.c_str(), best_name_len - 1);
    best_name[best_name_len - 1] = '\0';

    return AT_OK;
  }

  const char *at_status_str(at_status_t s)
  {
    switch (s)
    {
    case AT_OK:
      return "AT_OK";
    case AT_ERR_NO_MEMORY:
      return "AT_ERR_NO_MEMORY";
    case AT_ERR_INVALID_ARG:
      return "AT_ERR_INVALID_ARG";
    case AT_ERR_NOT_INITIALIZED:
      return "AT_ERR_NOT_INITIALIZED";
    case AT_ERR_NO_KERNELS:
      return "AT_ERR_NO_KERNELS";
    case AT_ERR_BENCHMARK_FAIL:
      return "AT_ERR_BENCHMARK_FAIL";
    case AT_ERR_CONFLICT:
      return "AT_ERR_CONFLICT";
    case AT_ERR_INTERNAL:
      return "AT_ERR_INTERNAL";
    default:
      return "Unknown AT error";
    }
  }

} // extern "C"