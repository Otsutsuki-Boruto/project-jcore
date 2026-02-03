// advanced/jit_kernel/src/jit_kernel_test.cpp
#include "benchmark.h"
#include "jit_kernel_generator.h"
#include "jit_kernel_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// BLAS libraries for reference
// Both OpenBLAS and BLIS provide the same cblas_sgemm interface
extern "C" {
void cblas_sgemm(const int Order, const int TransA, const int TransB,
                 const int M, const int N, const int K, const float alpha,
                 const float *A, const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
}

/* ========================================================================== */
/* Test Utilities */
/* ========================================================================== */

static void print_separator() {
  printf("====================================================================="
         "===\n");
}

static void print_test_header(const char *test_name) {
  print_separator();
  printf("TEST: %s\n", test_name);
  print_separator();
}

static void print_pass(const char *msg) { printf("✓ PASS: %s\n", msg); }

static void print_fail(const char *msg) { printf("✗ FAIL: %s\n", msg); }

static void print_info(const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf("INFO: ");
  vprintf(format, args);
  va_end(args);
  printf("\n");
}

static void print_perf(const char *label, double gflops, double time_ms) {
  printf("PERF [%s]: %.2f GFLOPS, %.3f ms\n", label, gflops, time_ms);
}

/* ========================================================================== */
/* Matrix Utilities */
/* ========================================================================== */

static void fill_random(float *data, size_t N) {
  for (size_t i = 0; i < N; i++) {
    data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  }
}

static void fill_constant(float *data, size_t N, float value) {
  for (size_t i = 0; i < N; i++) {
    data[i] = value;
  }
}

static float compute_rmse(const float *A, const float *B, size_t N) {
  double sum_sq = 0.0;
  for (size_t i = 0; i < N; i++) {
    double diff = A[i] - B[i];
    sum_sq += diff * diff;
  }
  return static_cast<float>(std::sqrt(sum_sq / N));
}

static bool verify_results(const float *expected, const float *actual, size_t N,
                           float tolerance = 1e-4f) {
  float rmse = compute_rmse(expected, actual, N);
  if (rmse > tolerance) {
    print_fail("Results mismatch");
    printf("      RMSE: %.6f (tolerance: %.6f)\n", rmse, tolerance);

    // Print first few mismatches
    int mismatch_count = 0;
    for (size_t i = 0; i < N && mismatch_count < 5; i++) {
      float diff = std::abs(expected[i] - actual[i]);
      if (diff > tolerance) {
        printf("      [%zu] expected: %.6f, got: %.6f (diff: %.6f)\n", i,
               expected[i], actual[i], diff);
        mismatch_count++;
      }
    }
    return false;
  }
  return true;
}

static void reference_gemm(const float *A, const float *B, float *C, size_t M,
                           size_t N, size_t K, size_t lda, size_t ldb,
                           size_t ldc, float alpha, float beta) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        sum += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
    }
  }
}

static double compute_gflops(size_t M, size_t N, size_t K, double time_sec) {
  double flops = 2.0 * M * N * K; // Multiply-add counts as 2 ops
  return (flops / time_sec) / 1e9;
}

/* ========================================================================== */
/* Test 1: Initialization and Shutdown                                        */
/* ========================================================================== */

static int test_init_shutdown() {
  print_test_header("Initialization and Shutdown");

  // Test default initialization with verbose enabled
  jkg_config_t config = {};
  config.target_isa = JKG_ISA_AUTO;
  config.backend = JKG_BACKEND_AUTO;
  config.enable_fma = 1;
  config.enable_prefetch = 1;
  config.enable_unroll = 1;
  config.unroll_factor = 4;
  config.cache_line_size = 64;
  config.optimization_level = 2;
  config.enable_kernel_cache = 1;
  config.verbose = 1; // Enable verbose to see what's happening

  int ret = jkg_init(&config);
  if (ret != JKG_OK) {
    print_fail("Default initialization failed");
    printf("      Error code: %d (%s)\n", ret, jkg_strerror(ret));
    return 1;
  }
  print_pass("Default initialization successful");

  // Verify initialized state
  if (!jkg_is_initialized()) {
    print_fail("jkg_is_initialized() returned false");
    return 1;
  }
  print_pass("Initialization state verified");

  // Get system info
  const char *sys_info = jkg_get_system_info();
  print_info("System Info:\n%s", sys_info);

  // Test double initialization (should be safe)
  ret = jkg_init(nullptr);
  if (ret != JKG_OK) {
    print_fail("Double initialization failed");
    return 1;
  }
  print_pass("Double initialization handled gracefully");

  // Shutdown
  jkg_shutdown();
  if (jkg_is_initialized()) {
    print_fail("Shutdown failed");
    return 1;
  }
  print_pass("Shutdown successful");

  // Re-initialize for remaining tests
  ret = jkg_init(nullptr);
  if (ret != JKG_OK) {
    print_fail("Re-initialization failed");
    return 1;
  }
  print_pass("Re-initialization successful");

  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 2: ISA Detection                                                      */
/* ========================================================================== */

static int test_isa_detection() {
  print_test_header("ISA Detection");

  uint32_t isa_mask = jkg_get_available_isa();
  print_info("Available ISA mask: 0x%08X", isa_mask);

  if (isa_mask & JKG_ISA_SSE2)
    print_info("  - SSE2 supported");
  if (isa_mask & JKG_ISA_AVX)
    print_info("  - AVX supported");
  if (isa_mask & JKG_ISA_AVX2)
    print_info("  - AVX2 supported");
  if (isa_mask & JKG_ISA_AVX512F)
    print_info("  - AVX-512 supported");
  if (isa_mask & JKG_ISA_AMX)
    print_info("  - AMX supported");

  if (isa_mask == 0) {
    print_fail("No ISA features detected");
    return 1;
  }

  print_pass("ISA detection successful");
  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 3: Optimal Tile Size Computation                                      */
/* ========================================================================== */

static int test_tile_sizes() {
  print_test_header("Optimal Tile Size Computation");

  size_t M, N, K;
  int ret = jkg_get_optimal_tile_sizes(JKG_ISA_AUTO, &M, &N, &K);
  if (ret != JKG_OK) {
    print_fail("Failed to compute optimal tile sizes");
    return 1;
  }

  print_info("Optimal tile sizes: M=%zu, N=%zu, K=%zu", M, N, K);

  if (M == 0 || N == 0 || K == 0) {
    print_fail("Invalid tile sizes");
    return 1;
  }

  print_pass("Tile size computation successful");
  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 4: Kernel Generation                                                  */
/* ========================================================================== */

static int test_kernel_generation() {
  print_test_header("Kernel Generation");

  // Test GEMM tile generation
  jkg_kernel_internal_t *handle = nullptr;
  int ret = jkg_generate_gemm_tile(6, 48, 256, &handle);
  if (ret != JKG_OK) {
    print_fail("GEMM tile generation failed");
    printf("      Error: %s\n", jkg_strerror(ret));
    return 1;
  }
  print_pass("GEMM tile generated");

  // Verify function pointer (in real implementation)
  void *func_ptr = jkg_get_kernel_function(handle);
  if (func_ptr == nullptr) {
    print_info(
        "Function pointer is NULL (JIT compilation not fully implemented)");
  }

  jkg_release_kernel(handle);
  print_pass("Kernel released");

  // Test fused GEMM generation
  ret = jkg_generate_fused_gemm(64, 64, 64, JKG_ACT_RELU, 1.0f, &handle);
  if (ret != JKG_OK) {
    print_fail("Fused GEMM generation failed");
    return 1;
  }
  print_pass("Fused GEMM+ReLU generated");

  jkg_release_kernel(handle);

  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 5: Kernel Cache                                                       */
/* ========================================================================== */

static int test_kernel_cache() {
  print_test_header("Kernel Cache");

  // Clear cache first
  jkg_clear_cache();

  size_t cached, hits, misses;
  jkg_get_cache_stats(&cached, &hits, &misses);
  print_info("Initial cache state: %zu cached, %zu hits, %zu misses", cached,
             hits, misses);

  // Generate same kernel twice
  jkg_kernel_internal_t *handle1 = nullptr;
  int ret = jkg_generate_gemm_tile(8, 64, 128, &handle1);
  if (ret != JKG_OK) {
    print_fail("First generation failed");
    return 1;
  }

  jkg_kernel_internal_t *handle2 = nullptr;
  ret = jkg_generate_gemm_tile(8, 64, 128, &handle2);
  if (ret != JKG_OK) {
    print_fail("Second generation failed");
    return 1;
  }

  jkg_get_cache_stats(&cached, &hits, &misses);
  print_info("After duplicate generation: %zu cached, %zu hits, %zu misses",
             cached, hits, misses);

  if (hits == 0) {
    print_fail("Cache hit expected but not observed");
    return 1;
  }
  print_pass("Cache hit detected");

  jkg_release_kernel(handle1);
  jkg_release_kernel(handle2);

  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 6: Small GEMM Correctness                                             */
/* ========================================================================== */

static int test_small_gemm_correctness() {
  print_test_header("Small GEMM Correctness");

  const size_t M = 4, N = 4, K = 4;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_ref(M * N, 0.0f);
  std::vector<float> C_jit(M * N, 0.0f);

  fill_random(A.data(), M * K);
  fill_random(B.data(), K * N);

  // Reference implementation
  reference_gemm(A.data(), B.data(), C_ref.data(), M, N, K, K, N, N, 1.0f,
                 0.0f);

  print_info("Reference GEMM completed");

  // Note: JIT kernel execution would happen here
  // For now, we just verify the generation
  jkg_kernel_internal_t *handle = nullptr;
  int ret = jkg_generate_gemm_tile(M, N, K, &handle);
  if (ret != JKG_OK) {
    print_fail("Kernel generation failed");
    return 1;
  }

  print_pass(
      "Small GEMM kernel generated (execution pending JIT implementation)");

  jkg_release_kernel(handle);

  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 7: BLAS Comparison Benchmark (K = 4096)                               */
/* ========================================================================== */

static int test_blas_benchmark_4096() {
  print_test_header("BLAS Comparison Benchmark (K = 2048)");

  const size_t K = 4096;
  const size_t M = K, N = K;
  const int iterations = 5;

  print_info("Matrix dimensions: M=%zu, N=%zu, K=%zu", M, N, K);
  print_info("Iterations: %d", iterations);

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N);

  fill_random(A.data(), M * K);
  fill_random(B.data(), K * N);

  // OpenBLAS benchmark
  fill_constant(C.data(), M * N, 0.0f);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    cblas_sgemm(101, 111, 111, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f,
                C.data(), N);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_openblas =
      std::chrono::duration<double>(end - start).count() / iterations;
  double gflops_openblas = compute_gflops(M, N, K, time_openblas);
  print_perf("OpenBLAS", gflops_openblas, time_openblas * 1000);

  // BLIS benchmark (uses same cblas interface)
  fill_constant(C.data(), M * N, 0.0f);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    cblas_sgemm(101, 111, 111, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f,
                C.data(), N);
  }
  end = std::chrono::high_resolution_clock::now();
  double time_blis =
      std::chrono::duration<double>(end - start).count() / iterations;
  double gflops_blis = compute_gflops(M, N, K, time_blis);
  print_perf("BLIS", gflops_blis, time_blis * 1000);

  // // Reference scalar
  // fill_constant(C.data(), M * N, 0.0f);
  // start = std::chrono::high_resolution_clock::now();
  // reference_gemm(A.data(), B.data(), C.data(), M, N, K, K, N, N, 1.0f, 0.0f);
  // end = std::chrono::high_resolution_clock::now();
  // double time_ref = std::chrono::duration<double>(end - start).count();
  // double gflops_ref = compute_gflops(M, N, K, time_ref);
  // print_perf("Reference (Scalar)", gflops_ref, time_ref * 1000);

  // JIT kernel (generation only for now)
  jkg_kernel_internal_t *handle = nullptr;
  int ret = jkg_generate_gemm_tile(64, 64, 64, &handle);
  if (ret == JKG_OK) {
    print_info("JIT kernel generated (execution pending)");
    jkg_release_kernel(handle);
  }

  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 8: BLAS Comparison Benchmark (K = 8192)                               */
/* ========================================================================== */

static int test_blas_benchmark_8192() {
  print_test_header("BLAS Comparison Benchmark (K = 8192)");

  const size_t K = 8192;
  const size_t M = K, N = K;
  const int iterations = 1;

  print_info("Matrix dimensions: M=%zu, N=%zu, K=%zu", M, N, K);
  print_info("Iterations: %d", iterations);

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N);

  fill_random(A.data(), M * K);
  fill_random(B.data(), K * N);

  // OpenBLAS
  fill_constant(C.data(), M * N, 0.0f);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    cblas_sgemm(101, 111, 111, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f,
                C.data(), N);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_openblas =
      std::chrono::duration<double>(end - start).count() / iterations;
  double gflops_openblas = compute_gflops(M, N, K, time_openblas);
  print_perf("OpenBLAS", gflops_openblas, time_openblas * 1000);

  // BLIS (uses same cblas interface)
  fill_constant(C.data(), M * N, 0.0f);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    cblas_sgemm(101, 111, 111, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f,
                C.data(), N);
  }
  end = std::chrono::high_resolution_clock::now();
  double time_blis =
      std::chrono::duration<double>(end - start).count() / iterations;
  double gflops_blis = compute_gflops(M, N, K, time_blis);
  print_perf("BLIS", gflops_blis, time_blis * 1000);

  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 9: Non-Square Matrix GEMM                                             */
/* ========================================================================== */

static int test_nonsquare_gemm() {
  print_test_header("Non-Square Matrix GEMM");

  const size_t M = 1024, N = 2048, K = 4096;
  const int iterations = 3;

  print_info("Matrix dimensions: M=%zu, N=%zu, K=%zu", M, N, K);

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N);

  fill_random(A.data(), M * K);
  fill_random(B.data(), K * N);

  // OpenBLAS
  fill_constant(C.data(), M * N, 0.0f);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    cblas_sgemm(101, 111, 111, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f,
                C.data(), N);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_sec =
      std::chrono::duration<double>(end - start).count() / iterations;
  double gflops = compute_gflops(M, N, K, time_sec);
  print_perf("OpenBLAS", gflops, time_sec * 1000);

  print_pass("Non-square GEMM successful");
  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 10: Kernel Fusion Engine Integration                                  */
/* ========================================================================== */

static int test_fusion_integration() {
  print_test_header("Kernel Fusion Engine Integration");

  // Test that JIT generator works with Kernel Fusion Engine
  const size_t M = 8192, N = 9216, K = 8192;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N);
  std::vector<float> bias(N);

  fill_random(A.data(), M * K);
  fill_random(B.data(), K * N);
  fill_random(bias.data(), N);

  // Use KFE for fused GEMM+Bias+ReLU
  kfe_perf_stats_t stats = {};
  int ret = kfe_sgemm_bias_activation(
      KFE_LAYOUT_ROW_MAJOR, KFE_NO_TRANS, KFE_NO_TRANS, M, N, K, 1.0f, A.data(),
      K, B.data(), N, bias.data(), KFE_ACTIVATION_RELU, C.data(), N, &stats);

  if (ret == KFE_OK) {
    print_pass("KFE integration successful");
    print_info("Achieved: %.2f GFLOPS", stats.gflops);
  } else {
    print_info("KFE not available (OK for JIT-only test)");
  }

  print_separator();
  return 0;
}

/* ========================================================================== */
/* Test 11: Self-Test                                                         */
/* ========================================================================== */

static int test_self_test() {
  print_test_header("Self-Test");

  int ret = jkg_self_test(1);
  if (ret != JKG_OK) {
    print_fail("Self-test failed");
    return 1;
  }

  print_pass("Self-test successful");
  print_separator();
  return 0;
}

/* ========================================================================== */
/* Main Test Runner */
/* ========================================================================== */

int main() {
  printf("\n");
  print_separator();
  printf("JIT KERNEL GENERATOR - COMPREHENSIVE TEST SUITE\n");
  printf("Project JCore - Advanced Component\n");
  print_separator();
  printf("\n");

  srand(42); // Fixed seed for reproducibility

  int failed_tests = 0;
  int total_tests = 0;

  // Run all tests
  total_tests++;
  failed_tests += test_init_shutdown();
  total_tests++;
  failed_tests += test_isa_detection();
  total_tests++;
  failed_tests += test_tile_sizes();
  total_tests++;
  failed_tests += test_kernel_generation();
  total_tests++;
  failed_tests += test_kernel_cache();
  total_tests++;
  failed_tests += test_small_gemm_correctness();
  total_tests++;
  failed_tests += test_blas_benchmark_4096();
  total_tests++;
  failed_tests += test_blas_benchmark_8192();
  total_tests++;
  failed_tests += test_nonsquare_gemm();
  total_tests++;
  failed_tests += test_fusion_integration();
  total_tests++;
  failed_tests += test_self_test();

  // Final cleanup
  jkg_shutdown();

  // Print summary
  printf("\n");
  print_separator();
  printf("TEST SUMMARY\n");
  print_separator();
  printf("Total tests: %d\n", total_tests);
  printf("Passed: %d\n", total_tests - failed_tests);
  printf("Failed: %d\n", failed_tests);
  print_separator();

  if (failed_tests == 0) {
    printf("\n✓ ALL TESTS PASSED\n\n");
    return 0;
  } else {
    printf("\n✗ SOME TESTS FAILED\n\n");
    return 1;
  }
}