// adaptive_tuner_core.cpp
// Core implementation of Adaptive Kernel Auto-Tuner
// Manages registration and benchmarking of matrix multiplication kernels

#include "adaptive_tuner.h"
#include "jcore_isa_dispatch.h"
#include "benchmark.h"
#include "global_thread_scheduler.h"
#include "mem_wrapper.h"
#include "ffm_cache_block.h"
#include "ffm_prefetch.h"
#include "cpu_info.h"
#include "config.h"
#include "pool_manager.h"

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cmath>

// Internal kernel registry entry
struct KernelEntry
{
  std::string name;
  jcore_features_t required_features;
  jcore_matmul_f32_fn fn;
};

// Global state for the tuner
struct TunerState
{
  bool initialized;
  std::vector<KernelEntry> kernels;
  jcore::global_thread::GlobalThreadScheduler scheduler;
  ffm_cache_info_t *cache_info;
  cpu_info_t cpu_info{};
  jcore::config::Config config;
  pm_t *pool_manager;

  TunerState() : initialized(false), cache_info(nullptr), pool_manager(nullptr) {}
};

static TunerState g_tuner;

// Initialize the adaptive tuner
extern "C" at_status_t at_init(void)
{
  if (g_tuner.initialized)
  {
    return AT_OK; // Already initialized
  }

  // Initialize memory allocator
  ffm_status_t mem_status = ffm_init(FFM_BACKEND_AUTO);
  if (mem_status != FFM_OK)
  {
    fprintf(stderr, "[AT] Failed to initialize memory allocator: %s\n",
            ffm_status_str(mem_status));
    return AT_ERR_INTERNAL;
  }

  // Detect CPU features
  g_tuner.cpu_info = detect_cpu_info();

  // Initialize thread scheduler with global thread scheduler
  jcore::global_thread::SchedulerResult sched_result = g_tuner.scheduler.Init(g_tuner.config);
  if (!sched_result.ok)
  {
    fprintf(stderr, "[AT] Thread scheduler init failed: %s\n",
            sched_result.message.c_str());
    ffm_shutdown();
    return AT_ERR_INTERNAL;
  }

  // Initialize cache info
  g_tuner.cache_info = ffm_cache_init();
  if (!g_tuner.cache_info)
  {
    fprintf(stderr, "[AT] Warning: Cache info unavailable, using defaults\n");
  }

  // Initialize memory pool manager
  size_t pool_size = 256 * 1024 * 1024; // 256 MB
  size_t chunk_size = 64 * 1024; // 64 KB chunks
  pm_status_t pm_status = pm_init(&g_tuner.pool_manager, pool_size, chunk_size, 0, -1);
  if (pm_status != PM_OK)
  {
    fprintf(stderr, "[AT] Warning: Pool manager init failed, using FFM allocator\n");
    g_tuner.pool_manager = nullptr;
  }

  g_tuner.initialized = true;
  fprintf(stdout, "[AT] Adaptive Tuner initialized successfully\n");
  fprintf(stdout, "[AT] CPU: %d cores, AVX2=%d, AVX512=%d, AMX=%d\n",
          g_tuner.cpu_info.cores,
          g_tuner.cpu_info.avx2,
          g_tuner.cpu_info.avx512,
          g_tuner.cpu_info.amx);
  fprintf(stdout, "[AT] Scheduler backend: %s with %zu threads\n",
          g_tuner.scheduler.BackendName().c_str(),
          g_tuner.scheduler.GetNumThreads());

  return AT_OK;
}

// Shutdown and cleanup
extern "C" void at_shutdown(void)
{
  if (!g_tuner.initialized)
  {
    return;
  }

  // Clear kernel registry
  g_tuner.kernels.clear();

  // Cleanup cache info
  if (g_tuner.cache_info)
  {
    ffm_cache_free(g_tuner.cache_info);
    g_tuner.cache_info = nullptr;
  }

  // Shutdown scheduler
  g_tuner.scheduler.Shutdown();

  // Shutdown pool manager
  if (g_tuner.pool_manager)
  {
    pm_shutdown(g_tuner.pool_manager);
    g_tuner.pool_manager = nullptr;
  }

  // Shutdown memory allocator
  ffm_shutdown();

  g_tuner.initialized = false;
  fprintf(stdout, "[AT] Adaptive Tuner shutdown complete\n");
}

// Register a matrix multiplication implementation
extern "C" at_status_t at_register_matmul_impl(
    const char *name,
    unsigned long long required_features,
    jcore_matmul_f32_fn fn)
{
  if (!g_tuner.initialized)
  {
    return AT_ERR_NOT_INITIALIZED;
  }

  if (!name || !fn)
  {
    return AT_ERR_INVALID_ARG;
  }

  // Check for duplicate names
  for (const auto &entry : g_tuner.kernels)
  {
    if (entry.name == name)
    {
      fprintf(stderr, "[AT] Kernel '%s' already registered\n", name);
      return AT_ERR_CONFLICT;
    }
  }

  KernelEntry entry;
  entry.name = name;
  entry.required_features = required_features;
  entry.fn = fn;

  g_tuner.kernels.push_back(entry);
  fprintf(stdout, "[AT] Registered kernel: %s (features: 0x%llx)\n",
          name, required_features);

  return AT_OK;
}

// Helper: Check if kernel's required features are satisfied
static bool kernel_is_compatible(const KernelEntry &entry)
{
  jcore_features_t host_features = jcore_get_host_features();

  // Check if all required features are present
  return (entry.required_features & host_features) == entry.required_features;
}

// Benchmark result for a single kernel
struct BenchmarkResult
{
  std::string name;
  double gflops;
  double mean_usec;
  bool success;

  BenchmarkResult() : gflops(0.0), mean_usec(0.0), success(false) {}
};

// Benchmark a single kernel
static BenchmarkResult benchmark_kernel(
    const KernelEntry &entry,
    size_t M, size_t N, size_t K,
    size_t num_threads)
{
  BenchmarkResult result;
  result.name = entry.name;
  result.success = false;

  // Set BLIS threads via environment variable for the kernel
  size_t threads_to_use = (num_threads > 0) ? num_threads : g_tuner.scheduler.GetNumThreads();
  if (threads_to_use > 0)
  {
    char thread_str[32];
    snprintf(thread_str, sizeof(thread_str), "%zu", threads_to_use);
    setenv("BLIS_NUM_THREADS", "1", 1);
  }

  // Allocate aligned matrices
  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  float *A = g_tuner.pool_manager ? static_cast<float *>(pm_alloc(g_tuner.pool_manager)) : static_cast<float *>(ffm_aligned_alloc(64, size_A));
  float *B = g_tuner.pool_manager ? static_cast<float *>(pm_alloc(g_tuner.pool_manager)) : static_cast<float *>(ffm_aligned_alloc(64, size_B));
  float *C = g_tuner.pool_manager ? static_cast<float *>(pm_alloc(g_tuner.pool_manager)) : static_cast<float *>(ffm_aligned_alloc(64, size_C));

  if (!A || !B || !C)
  {
    fprintf(stderr, "[AT] Memory allocation failed for kernel '%s'\n",
            entry.name.c_str());
    if (A)
    {
      if (g_tuner.pool_manager) pm_free(g_tuner.pool_manager, A);
      else ffm_free(A);
    }
    if (B)
    {
      if (g_tuner.pool_manager) pm_free(g_tuner.pool_manager, B);
      else ffm_free(B);
    }
    if (C)
    {
      if (g_tuner.pool_manager) pm_free(g_tuner.pool_manager, C);
      else ffm_free(C);
    }
    return result;
  }

  // Initialize matrices with random data
  for (size_t i = 0; i < M * K; ++i)
  {
    A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  for (size_t i = 0; i < K * N; ++i)
  {
    B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  for (size_t i = 0; i < M * N; ++i)
  {
    C[i] = 0.0f;
  }

  // Prefetch input matrices to ensure consistent cache state before benchmarking
  ffm_prefetch_block_read_T0(A, size_A);
  ffm_prefetch_block_read_T0(B, size_B);
  ffm_prefetch_block_write_T0(C, size_C);

  // Configure benchmark options

  // Configure benchmark options
  jcore::microbench::RunOptions opts;
  opts.warmup_iterations = 3;
  opts.iterations = 10;
  opts.samples = 5;
  opts.capture_cycles = false;

  std::vector<jcore::microbench::Sample> samples;

  // Run benchmark
  bool bench_ok = jcore::microbench::RunMicrobenchmark(
      [&](size_t)
      {
        entry.fn(A, B, C, M, N, K);
      },
      opts, samples);

  if (!bench_ok || samples.empty())
  {
    fprintf(stderr, "[AT] Benchmark failed for kernel '%s'\n",
            entry.name.c_str());
    if (g_tuner.pool_manager)
    {
      pm_free(g_tuner.pool_manager, A);
      pm_free(g_tuner.pool_manager, B);
      pm_free(g_tuner.pool_manager, C);
    }
    else
    {
      ffm_free(A);
      ffm_free(B);
      ffm_free(C);
    }
    return result;
  }

  // Compute summary statistics
  jcore::microbench::Summary summary;
  if (!jcore::microbench::Summarize(samples, summary))
  {
    fprintf(stderr, "[AT] Failed to compute statistics for '%s'\n",
            entry.name.c_str());
    if (g_tuner.pool_manager)
    {
      pm_free(g_tuner.pool_manager, A);
      pm_free(g_tuner.pool_manager, B);
      pm_free(g_tuner.pool_manager, C);
    }
    else
    {
      ffm_free(A);
      ffm_free(B);
      ffm_free(C);
    }
    return result;
  }

  // Calculate GFLOPS: 2*M*N*K operations per matmul
  double ops = 2.0 * M * N * K;
  double mean_sec = summary.mean_usec / 1e6;
  double gflops = (ops / mean_sec) / 1e9;

  result.gflops = gflops;
  result.mean_usec = summary.mean_usec;
  result.success = true;

  fprintf(stdout, "[AT] %s: %.2f GFLOPS (%.2f us)\n",
          entry.name.c_str(), gflops, summary.mean_usec);

  // Cleanup
  if (g_tuner.pool_manager)
  {
    pm_free(g_tuner.pool_manager, A);
    pm_free(g_tuner.pool_manager, B);
    pm_free(g_tuner.pool_manager, C);
  }
  else
  {
    ffm_free(A);
    ffm_free(B);
    ffm_free(C);
  }

  return result;
}

// Benchmark all registered kernels and return the best one
extern "C" at_status_t at_benchmark_matmul_all(
    size_t M, size_t N, size_t K,
    size_t preferred_threads,
    size_t tile_size_hint,
    char *best_name,
    size_t best_name_len)
{
  if (!g_tuner.initialized)
  {
    return AT_ERR_NOT_INITIALIZED;
  }

  if (M == 0 || N == 0 || K == 0)
  {
    return AT_ERR_INVALID_ARG;
  }

  if (!best_name || best_name_len == 0)
  {
    return AT_ERR_INVALID_ARG;
  }

  if (g_tuner.kernels.empty())
  {
    return AT_ERR_NO_KERNELS;
  }

  fprintf(stdout, "[AT] ======================================\n");
  fprintf(stdout, "[AT] Benchmarking kernels for M=%zu N=%zu K=%zu\n",
          M, N, K);
  fprintf(stdout, "[AT] Threads: %zu, Tile hint: %zu\n",
          preferred_threads, tile_size_hint);
  fprintf(stdout, "[AT] ======================================\n");

  // Filter compatible kernels
  std::vector<KernelEntry> compatible;
  for (const auto &entry : g_tuner.kernels)
  {
    if (kernel_is_compatible(entry))
    {
      compatible.push_back(entry);
    }
    else
    {
      fprintf(stdout, "[AT] Skipping incompatible kernel: %s\n",
              entry.name.c_str());
    }
  }

  if (compatible.empty())
  {
    fprintf(stderr, "[AT] No compatible kernels available\n");
    return AT_ERR_NO_KERNELS;
  }

  // Benchmark each compatible kernel
  std::vector<BenchmarkResult> results;
  for (const auto &entry : compatible)
  {
    BenchmarkResult res = benchmark_kernel(entry, M, N, K, preferred_threads);
    if (res.success)
    {
      results.push_back(res);
    }
  }

  if (results.empty())
  {
    fprintf(stderr, "[AT] All benchmarks failed\n");
    return AT_ERR_BENCHMARK_FAIL;
  }

  // Find the best performer (highest GFLOPS)
  auto best = std::max_element(
      results.begin(), results.end(),
      [](const BenchmarkResult &a, const BenchmarkResult &b)
      {
        return a.gflops < b.gflops;
      });

  // Copy best kernel name to output buffer
  size_t copy_len = std::min(best->name.size(), best_name_len - 1);
  std::memcpy(best_name, best->name.c_str(), copy_len);
  best_name[copy_len] = '\0';

  fprintf(stdout, "[AT] ======================================\n");
  fprintf(stdout, "[AT] WINNER: %s with %.2f GFLOPS\n",
          best->name.c_str(), best->gflops);
  fprintf(stdout, "[AT] ======================================\n");

  return AT_OK;
}

// Convert status code to string
extern "C" const char *at_status_str(at_status_t s)
{
  switch (s)
  {
  case AT_OK:
    return "Success";
  case AT_ERR_NO_MEMORY:
    return "Out of memory";
  case AT_ERR_INVALID_ARG:
    return "Invalid argument";
  case AT_ERR_NOT_INITIALIZED:
    return "Not initialized";
  case AT_ERR_NO_KERNELS:
    return "No kernels registered";
  case AT_ERR_BENCHMARK_FAIL:
    return "Benchmark failed";
  case AT_ERR_CONFLICT:
    return "Conflict (duplicate name)";
  case AT_ERR_INTERNAL:
    return "Internal error";
  default:
    return "Unknown error";
  }
}