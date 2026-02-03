#include "k_kernel_dispatch.h"
#include "ffm_prefetch.h"
#include "k_thread_verify.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <xmmintrin.h>

/* Internal state */
namespace {
static std::map<std::tuple<size_t, size_t, size_t>, std::string>
    best_kernel_registry;

std::mutex g_init_mutex;
bool g_initialized = false;

std::shared_mutex g_kernel_mutex;
std::unordered_map<std::string, jcore_matmul_f32_fn> g_kernel_map;

// Forward declarations
extern "C" at_status_t at_benchmark_matmul_all(size_t M, size_t N, size_t K,
                                               size_t preferred_threads,
                                               size_t tile_size_hint,
                                               char *best_name,
                                               size_t best_name_len);
extern "C" const char *at_status_str(at_status_t s);

// Strategic prefetch: only prefetch first few cache lines of each matrix
inline void strategic_prefetch_tiled(const void *ptr, size_t rows, size_t cols,
                                     size_t elem_size, size_t tile_size) {
  if (!ptr || rows == 0 || cols == 0)
    return;

  const char *base = static_cast<const char *>(ptr);
  size_t row_bytes = cols * elem_size;
  size_t prefetch_rows = std::min(rows, tile_size);
  size_t prefetch_bytes = std::min(row_bytes, tile_size * elem_size);

  // SSE prefetch for first tile
  for (size_t i = 0; i < prefetch_rows; ++i) {
    const char *row_start = base + i * row_bytes;
    for (size_t j = 0; j < prefetch_bytes; j += 64) {
      _mm_prefetch(row_start + j, _MM_HINT_T0);
    }
  }
}

// Core kernel picker (per shape)
jcore_matmul_f32_fn pick_best_kernel(size_t M, size_t N, size_t K) {
  if (!g_initialized)
    return nullptr;

  auto key = std::make_tuple(M, N, K);

  // Check cache (shared lock)
  {
    std::shared_lock<std::shared_mutex> read_lock(g_kernel_mutex);
    auto it = best_kernel_registry.find(key);
    if (it != best_kernel_registry.end()) {
      auto found = g_kernel_map.find(it->second);
      if (found != g_kernel_map.end())
        return found->second;
    }
  }

  // Select best kernel via tuner
  char best[128] = {0};
  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0)
    hw_threads = 1;

  // Compute adaptive thread count with large matrix override
  size_t work_size = M * N * K;
  size_t adaptive_threads = hw_threads; // default to all threads

  // Only reduce threads for small matrices
  if (work_size < 100000000UL) { // 100M elements
    size_t min_work_per_thread = 1024 * 1024;
    adaptive_threads = std::max(1UL, std::min(static_cast<size_t>(hw_threads),
                                              work_size / min_work_per_thread));
  }

  at_status_t s =
      at_benchmark_matmul_all(M, N, K, adaptive_threads, 0, best, sizeof(best));

  if (s != AT_OK) {
    std::cerr << "[DISPATCH] Adaptive tuner failed: " << at_status_str(s)
              << "\n";
    return nullptr;
  }

  std::string best_kernel(best);
  std::cerr << "\n[DISPATCH] Selected kernel for [" << M << "x" << N << "x" << K
            << "]: " << best_kernel << "\n\n";

  // Store result in registry (write lock)
  {
    std::unique_lock<std::shared_mutex> write_lock(g_kernel_mutex);
    best_kernel_registry[key] = best_kernel;
  }

  // Return function pointer
  auto it = g_kernel_map.find(best_kernel);
  if (it != g_kernel_map.end())
    return it->second;

  return nullptr;
}

// Threading verification helper
void verify_threading_setup() {
  unsigned int hw_threads = std::thread::hardware_concurrency();

  const char *blis_env = std::getenv("BLIS_NUM_THREADS");
  const char *openblas_env = std::getenv("OPENBLAS_NUM_THREADS");
  const char *omp_env = std::getenv("OMP_NUM_THREADS");

  std::cerr << "\n[DISPATCH] Threading verification:\n";
  std::cerr << "  Hardware threads: " << hw_threads << "\n";
  std::cerr << "  BLIS_NUM_THREADS: " << (blis_env ? blis_env : "not set")
            << "\n";
  std::cerr << "  OPENBLAS_NUM_THREADS: "
            << (openblas_env ? openblas_env : "not set") << "\n";
  std::cerr << "  OMP_NUM_THREADS: " << (omp_env ? omp_env : "not set") << "\n";
}

} // namespace

extern "C" {

int k_dispatch_init(void) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized)
    return JCORE_OK;

  if (at_init() != AT_OK) {
    std::cerr << "[DISPATCH] Adaptive tuner init failed!\n";
    return JCORE_ERR_INTERNAL;
  }

  // Threading environment setup with persistent strings
  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0)
    hw_threads = 4; // conservative fallback

  // Use static strings to avoid dangling pointer issues
  static std::string blis_threads_str = std::to_string(hw_threads);
  static std::string openblas_threads_str = std::to_string(hw_threads);
  static std::string omp_threads_str = std::to_string(hw_threads);

  // Set threading for BLIS (uses OpenMP typically)
  setenv("BLIS_NUM_THREADS", blis_threads_str.c_str(), 1);

  // Set threading for OpenBLAS (multiple possible env vars)
  setenv("OPENBLAS_NUM_THREADS", openblas_threads_str.c_str(), 1);
  setenv("GOTO_NUM_THREADS", openblas_threads_str.c_str(), 1); // legacy

  // Optional: if you want to try TBB for MKL (if present)
  // Note: OpenBLAS typically uses pthreads or OpenMP, not TBB
  // Only set this if you know your build supports it
  // setenv("MKL_THREADING_LAYER", "GNU", 1); // Use OpenMP for MKL if present

  // Verify threading setup
  verify_threading_setup();

  // Register available kernels
  void blis_sgemm(const float *, const float *, float *, size_t, size_t,
                  size_t);
  void openblas_sgemm(const float *, const float *, float *, size_t, size_t,
                      size_t);
  void blis_micro_sgemm(const float* A, const float* B, float* C,
                                 size_t M, size_t N, size_t K);

  {
    std::unique_lock<std::shared_mutex> write_lock(g_kernel_mutex);
    g_kernel_map["openblas_sgemm"] = openblas_sgemm;
    g_kernel_map["blis_sgemm"] = blis_sgemm;
  }

  // Register kernels with correct feature requirements
  jcore_features_t host_feat = jcore_get_host_features();

  // OpenBLAS: requires at minimum AVX, benefits from AVX2/AVX512
  jcore_features_t openblas_feat = JCORE_FEAT_AVX;
  if (host_feat & JCORE_FEAT_AVX2)
    openblas_feat |= JCORE_FEAT_AVX2;
  if (host_feat & JCORE_FEAT_AVX512)
    openblas_feat |= JCORE_FEAT_AVX512;

  // BLIS: requires AVX minimum
  jcore_features_t blis_feat = JCORE_FEAT_AVX;
  if (host_feat & JCORE_FEAT_AVX2)
    blis_feat |= JCORE_FEAT_AVX2;

  at_register_matmul_impl("openblas_sgemm", openblas_feat, openblas_sgemm);
  at_register_matmul_impl("blis_sgemm", blis_feat, blis_sgemm);

  std::cerr << "[DISPATCH] Initialization complete. Registered 2 kernels.\n";

  g_initialized = true;
  return JCORE_OK;
}

void k_dispatch_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (!g_initialized)
    return;

  {
    std::unique_lock<std::shared_mutex> write_lock(g_kernel_mutex);
    best_kernel_registry.clear();
    g_kernel_map.clear();
  }

  at_shutdown();
  g_initialized = false;

  std::cerr << "[DISPATCH] Shutdown complete.\n";
}

int k_dispatch_matmul(const float *A, const float *B, float *C, size_t M,
                      size_t N, size_t K) {
  if (!A || !B || !C)
    return JCORE_ERR_INVALID_ARG;

  if (!g_initialized)
    return JCORE_ERR_NOT_INITIALIZED;

  // Strategic prefetch: prime hardware prefetcher with first few cache lines
  // This is enough to trigger streaming prefetch in modern CPUs
  // Avoid massive bulk prefetch that causes cache pollution
  size_t tile = std::min({M, N, K, 128UL});
  strategic_prefetch_tiled(A, M, K, sizeof(float), tile);
  strategic_prefetch_tiled(B, K, N, sizeof(float), tile);

  jcore_matmul_f32_fn fn = pick_best_kernel(M, N, K);
  if (!fn) {
    // Fallback to scalar implementation
    extern void jcore_scalar_matmul_impl(void *);
    jcore_matmul_args_t args{A, B, C, M, N, K};
    jcore_scalar_matmul_impl(&args);
    return JCORE_OK;
  }

  fn(A, B, C, M, N, K);
  return JCORE_OK;
}

const char *k_dispatch_get_last_selected_kernel() {
  std::shared_lock<std::shared_mutex> read_lock(g_kernel_mutex);
  if (!best_kernel_registry.empty()) {
    auto it = best_kernel_registry.rbegin();
    return it->second.c_str();
  }
  return "unknown";
}

} // extern "C"