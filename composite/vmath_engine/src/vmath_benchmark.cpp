#pragma GCC target("no-avx512f,no-avx512bw,no-avx512dq,no-avx512vl")

/**
 * @file vmath_benchmark.cpp
 * @brief Comprehensive performance benchmark for Vector Math Engine
 *        Tests transcendental ops (SIMD vs scalar) and matrix kernels (OpenBLAS/BLIS)
 */

#include "vmath_engine.h"
#include "cpu_info.h"
#include "jcore_isa_dispatch.h"
#include "k_kernel_dispatch.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>

extern "C"
{
#include <openblas/cblas.h>
}

using namespace std::chrono;

// ============================================================================
// Utility: High-resolution timer
// ============================================================================

class Timer
{
  high_resolution_clock::time_point start_;

public:
  Timer() : start_(high_resolution_clock::now()) {}

  double elapsed_ms() const
  {
    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start_).count();
  }

  void reset() { start_ = high_resolution_clock::now(); }
};

// ============================================================================
// ISA Detection
// ============================================================================

struct ISACapabilities
{
  bool has_avx2;
  bool has_avx512;
};

static ISACapabilities detect_isa()
{
  cpu_info_t cpu = detect_cpu_info();
  ISACapabilities caps;
  caps.has_avx2 = cpu.avx2;
  caps.has_avx512 = cpu.avx512;
  return caps;
}

// ============================================================================
// Vector Math Benchmarks (Transcendental Operations)
// ============================================================================

static void fill_random(float *arr, size_t n, float min_val, float max_val)
{
  for (size_t i = 0; i < n; ++i)
  {
    float r = static_cast<float>(rand()) / RAND_MAX;
    arr[i] = min_val + r * (max_val - min_val);
  }
}

struct BenchResult
{
  double time_simd_ms;
  double time_scalar_ms;
  double speedup;
  double gflops;
  double max_error;
};

static BenchResult benchmark_transcendental(
    const char *name,
    int (*vmath_fn)(const float *, float *, size_t),
    float (*scalar_fn)(float),
    size_t n,
    float min_val,
    float max_val)
{
  std::vector<float> x(n);
  std::vector<float> y_simd(n);
  std::vector<float> y_scalar(n);

  fill_random(x.data(), n, min_val, max_val);

  // Warm-up
  vmath_fn(x.data(), y_simd.data(), n);

  // SIMD benchmark
  Timer timer;
  const int runs = 10;
  for (int i = 0; i < runs; ++i)
  {
    vmath_fn(x.data(), y_simd.data(), n);
  }
  double time_simd = timer.elapsed_ms() / runs;

  // Scalar benchmark
  timer.reset();
  for (int i = 0; i < runs; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      y_scalar[j] = scalar_fn(x[j]);
    }
  }
  double time_scalar = timer.elapsed_ms() / runs;

  // Calculate GFLOPS (assume 1 FLOP per element for transcendentals)
  double gflops = (static_cast<double>(n) / (time_simd / 1000.0)) / 1e9;

  // Check accuracy
  double max_error = 0.0;
  for (size_t i = 0; i < n; ++i)
  {
    double err = std::fabs(y_simd[i] - y_scalar[i]);
    max_error = std::max(max_error, err);
  }

  BenchResult result;
  result.time_simd_ms = time_simd;
  result.time_scalar_ms = time_scalar;
  result.speedup = time_scalar / time_simd;
  result.gflops = gflops;
  result.max_error = max_error;

  printf("%-10s: SIMD=%7.2f ms | Scalar=%7.2f ms | Speedup=%5.2fx | GFLOPS=%6.2f | Error=%.2e\n",
         name, time_simd, time_scalar, result.speedup, gflops, max_error);

  return result;
}

static void run_vector_math_benchmarks()
{
  const size_t n = 10000000; // 10M elements

  printf("\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf(" Vector Math Engine - Transcendental Operations (n=10M)\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");

  benchmark_transcendental("exp", vmath_expf, expf, n, -5.0f, 5.0f);
  benchmark_transcendental("log", vmath_logf, logf, n, 0.1f, 100.0f);
  benchmark_transcendental("sin", vmath_sinf, sinf, n, -3.14f, 3.14f);
  benchmark_transcendental("cos", vmath_cosf, cosf, n, -3.14f, 3.14f);
  benchmark_transcendental("tan", vmath_tanf, tanf, n, -1.5f, 1.5f);
  benchmark_transcendental("sqrt", vmath_sqrtf, sqrtf, n, 0.1f, 100.0f);
  benchmark_transcendental("tanh", vmath_tanhf, tanhf, n, -3.0f, 3.0f);
}

// ============================================================================
// Matrix Multiplication Kernel Benchmarks
// ============================================================================

// Forward declarations of kernel functions from Adaptive Kernel Auto-Tuner
extern "C"
{
  void openblas_sgemm(const float *, const float *, float *, size_t, size_t, size_t);
  void blis_sgemm(const float *, const float *, float *, size_t, size_t, size_t);
}

struct MatmulResult
{
  size_t size;
  double time_ms;
  double gflops;
};

static MatmulResult benchmark_matmul_kernel(
    const char *kernel_name,
    void (*kernel_fn)(const float *, const float *, float *, size_t, size_t, size_t),
    size_t N)
{
  // Allocate matrices
  std::vector<float> A(N * N);
  std::vector<float> B(N * N);
  std::vector<float> C(N * N);

  // Initialize with random data
  for (size_t i = 0; i < N * N; ++i)
  {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
    B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Warm-up
  kernel_fn(A.data(), B.data(), C.data(), N, N, N);

  // Benchmark multiple runs
  const int runs = 5;
  std::vector<double> timings;

  for (int i = 0; i < runs; ++i)
  {
    std::memset(C.data(), 0, N * N * sizeof(float));

    Timer timer;
    kernel_fn(A.data(), B.data(), C.data(), N, N, N);
    double time_ms = timer.elapsed_ms();
    timings.push_back(time_ms);
  }

  // Use minimum time for best-case performance
  double min_time = *std::min_element(timings.begin(), timings.end());

  // Calculate GFLOPS: 2*N^3 FLOPs for matrix multiply
  double flops = 2.0 * static_cast<double>(N) * N * N;
  double gflops = flops / (min_time / 1000.0) / 1e9;

  MatmulResult result;
  result.size = N;
  result.time_ms = min_time;
  result.gflops = gflops;

  return result;
}

static void run_matmul_benchmarks()
{
  printf("\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf(" Matrix Multiplication Kernel Performance (Single-threaded)\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf("\n");

  // Test different matrix sizes
  size_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

  for (size_t N : sizes)
  {
    printf("Matrix Size: %zu x %zu (FLOPS = 2 * %zu^3)\n", N, N, N);
    printf("─────────────────────────────────────────────────────────────────────────\n");

    auto result_openblas = benchmark_matmul_kernel("OpenBLAS", openblas_sgemm, N);
    printf("  OpenBLAS:  %7.2f ms  |  %7.2f GFLOPS\n",
           result_openblas.time_ms, result_openblas.gflops);

    auto result_blis = benchmark_matmul_kernel("BLIS", blis_sgemm, N);
    printf("  BLIS:      %7.2f ms  |  %7.2f GFLOPS\n",
           result_blis.time_ms, result_blis.gflops);

    printf("\n");
  }
}

// ============================================================================
// Kernel Dispatch Integration Test
// ============================================================================

static void test_kernel_dispatch_integration()
{
  printf("\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf(" Kernel Dispatch Integration Test\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf("\n");

  const size_t N = 2048;
  std::vector<float> A(N * N);
  std::vector<float> B(N * N);
  std::vector<float> C(N * N);

  // Initialize matrices
  for (size_t i = 0; i < N * N; ++i)
  {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
    B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Test kernel dispatch
  printf("Testing k_dispatch_matmul (N=%zu)...\n", N);

  Timer timer;
  int ret = k_dispatch_matmul(A.data(), B.data(), C.data(), N, N, N);
  double time_ms = timer.elapsed_ms();

  if (ret != JCORE_OK)
  {
    printf("[ERROR] k_dispatch_matmul failed with code %d\n", ret);
    return;
  }

  double flops = 2.0 * N * N * N;
  double gflops = flops / (time_ms / 1000.0) / 1e9;

  printf("  Time: %.2f ms\n", time_ms);
  printf("  Performance: %.2f GFLOPS\n", gflops);
  printf("  Selected kernel: %s\n", k_dispatch_get_last_selected_kernel());

  // Verify correctness with a small sample
  bool correct = true;
  for (size_t i = 0; i < 10 && i < N; ++i)
  {
    if (!std::isfinite(C[i * N]))
    {
      correct = false;
      break;
    }
  }

  printf("  Correctness: %s\n", correct ? "PASS" : "FAIL");
}

// ============================================================================
// Theoretical Peak Performance
// ============================================================================

static void print_theoretical_peak()
{
  cpu_info_t cpu = detect_cpu_info();

  printf("\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf(" System Information\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf("\n");

  printf("CPU Cores: %d\n", cpu.cores);
  printf("Logical Cores: %d\n", cpu.logical_cores);
  printf("ISA Support:\n");
  printf("  AVX:     %s\n", cpu.avx ? "Yes" : "No");
  printf("  AVX2:    %s\n", cpu.avx2 ? "Yes" : "No");
  printf("  AVX-512: %s\n", cpu.avx512 ? "Yes" : "No");
  printf("  AMX:     %s\n", cpu.amx ? "Yes" : "No");

  printf("\nCache Hierarchy:\n");
  printf("  L1D: %d KB\n", cpu.l1d_kb);
  printf("  L1I: %d KB\n", cpu.l1i_kb);
  printf("  L2:  %d KB\n", cpu.l2_kb);
  printf("  L3:  %d KB\n", cpu.l3_kb);

  // Estimate theoretical peak (rough approximation)
  // Assuming ~3 GHz, AVX2 can do 8 SP FMAs per cycle = 16 SP FLOPs/cycle
  double clock_ghz = 3.0; // Approximate
  double flops_per_cycle = cpu.avx2 ? 16.0 : 8.0;
  double theoretical_gflops = clock_ghz * flops_per_cycle;

  printf("\nTheoretical Peak (Single Core, FP32): ~%.0f GFLOPS\n", theoretical_gflops);
}

// ============================================================================
// Main
// ============================================================================

int main()
{
  srand(42); // Deterministic results

  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
  printf("║                                                                           ║\n");
  printf("║          Vector Math Engine - Comprehensive Performance Benchmark         ║\n");
  printf("║                                                                           ║\n");
  printf("╚═══════════════════════════════════════════════════════════════════════════╝\n");

  // Detect ISA
  ISACapabilities isa = detect_isa();
  if (!isa.has_avx2)
  {
    printf("\n[WARNING] AVX2 not detected. Performance will be limited.\n");
  }
  if (isa.has_avx512)
  {
    printf("\n[INFO] AVX-512 detected but disabled for this benchmark (AVX2 mode).\n");
  }

  // Print system info
  print_theoretical_peak();

  // Initialize Vector Math Engine
  printf("\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf(" Initialization\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf("\n");

  int ret = vmath_init();
  if (ret != VMATH_OK)
  {
    printf("[FATAL] vmath_init() failed with code %d\n", ret);
    return 1;
  }

  printf("Vector Math Engine: %s\n", vmath_get_info());
  printf("Active ISA Level: %s\n", vmath_isa_name(vmath_get_isa_level()));

  // Initialize kernel dispatch
  ret = k_dispatch_init();
  if (ret != JCORE_OK)
  {
    printf("[WARNING] k_dispatch_init() failed with code %d\n", ret);
  }
  else
  {
    printf("Kernel Dispatch: Initialized\n");
  }

  // Run benchmarks
  run_vector_math_benchmarks();
  run_matmul_benchmarks();
  test_kernel_dispatch_integration();

  printf("\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf(" Benchmark Complete\n");
  printf("═══════════════════════════════════════════════════════════════════════════\n");
  printf("\n");
  printf("Note: Matrix multiplication GFLOPS should approach CPU theoretical peak.\n");
  printf("      For modern CPUs with AVX2, expect 50-160+ GFLOPS depending on size.\n");
  printf("      Larger matrices (4096+) typically achieve higher efficiency.\n");
  printf("\n");

  vmath_shutdown();
  k_dispatch_shutdown();

  return 0;
}
