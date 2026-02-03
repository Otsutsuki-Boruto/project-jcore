#include "microkernel_interface.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <memory>

/* ========================================================================== */
/* Test Configuration                                                          */
/* ========================================================================== */

constexpr size_t TEST_SMALL_SIZE = 64;
constexpr size_t TEST_MEDIUM_SIZE = 256;
constexpr size_t TEST_LARGE_SIZE = 1024;
constexpr size_t TEST_XLARGE_SIZE = 2048;
constexpr size_t TEST_XXLARGE_SIZE = 4096;
constexpr size_t TEST_XXXLARGE_SIZE = 8192;
constexpr int TEST_ITERATIONS = 5;
constexpr int TEST_LARGE_ITERATIONS = 3;

constexpr const char *ANSI_COLOR_RED = "\x1b[31m";
constexpr const char *ANSI_COLOR_GREEN = "\x1b[32m";
constexpr const char *ANSI_COLOR_YELLOW = "\x1b[33m";
constexpr const char *ANSI_COLOR_BLUE = "\x1b[34m";
constexpr const char *ANSI_COLOR_MAGENTA = "\x1b[35m";
constexpr const char *ANSI_COLOR_CYAN = "\x1b[36m";
constexpr const char *ANSI_COLOR_RESET = "\x1b[0m";

/* ========================================================================== */
/* Test Statistics                                                             */
/* ========================================================================== */

struct TestStats
{
  int total_tests;
  int passed_tests;
  int failed_tests;
  int skipped_tests;
  double total_time_ms;

  TestStats() : total_tests(0), passed_tests(0), failed_tests(0),
                skipped_tests(0), total_time_ms(0.0) {}
};

static TestStats g_test_stats;

/* ========================================================================== */
/* Helper Functions                                                            */
/* ========================================================================== */

static void print_separator(char c)
{
  for (int i = 0; i < 80; ++i)
    std::printf("%c", c);
  std::printf("\n");
}

static void print_test_header(const char *test_name)
{
  std::printf("\n");
  print_separator('=');
  std::printf("%sTEST: %s%s\n", ANSI_COLOR_CYAN, test_name, ANSI_COLOR_RESET);
  print_separator('=');
}

static void print_test_result(const char *subtest, bool passed, const char *message = nullptr)
{
  g_test_stats.total_tests++;
  if (passed)
  {
    g_test_stats.passed_tests++;
    std::printf("%s[PASS]%s %s", ANSI_COLOR_GREEN, ANSI_COLOR_RESET, subtest);
    if (message)
      std::printf(" - %s", message);
    std::printf("\n");
  }
  else
  {
    g_test_stats.failed_tests++;
    std::printf("%s[FAIL]%s %s", ANSI_COLOR_RED, ANSI_COLOR_RESET, subtest);
    if (message)
      std::printf(" - %s", message);
    std::printf("\n");
  }
}

static void print_perf_stats(const mil_perf_stats_t *stats)
{
  if (stats == nullptr)
    return;

  std::printf("%s", ANSI_COLOR_YELLOW);
  std::printf("  Performance Statistics:\n");
  std::printf("    GFLOPS:     %.2f\n", stats->gflops);
  std::printf("    Time:       %.3f ms\n", stats->elapsed_ms);
  std::printf("    Bandwidth:  %.2f GB/s\n", stats->bandwidth_gbps);
  std::printf("    Kernel:     %s\n", stats->kernel_used);
  std::printf("    Backend:    %s\n", mil_backend_name(stats->backend_used));
  std::printf("%s", ANSI_COLOR_RESET);
  g_test_stats.total_time_ms += stats->elapsed_ms;
}

// RAII wrapper for matrix allocation with automatic cleanup
template <typename T>
struct Matrix {
  T* data;
  size_t size;

  Matrix(size_t rows, size_t cols, T init_val) : data(nullptr), size(rows * cols) {
    data = static_cast<T*>(std::malloc(size * sizeof(T)));
    if (data) {
      for (size_t i = 0; i < size; ++i) {
        if (init_val < 0) {
          data[i] = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX);
        } else {
          data[i] = init_val;
        }
      }
    }
  }

  ~Matrix() {
    if (data) {
      std::free(data);
      data = nullptr;
    }
  }

  // Delete copy constructor and assignment to prevent double-free
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  // Allow move semantics
  Matrix(Matrix&& other) noexcept : data(other.data), size(other.size) {
    other.data = nullptr;
    other.size = 0;
  }

  operator T*() { return data; }
  operator const T*() const { return data; }
  bool valid() const { return data != nullptr; }
};

template <typename T>
static bool verify_matrix_value(const T *matrix, size_t size, T expected, T tolerance)
{
  for (size_t i = 0; i < size; ++i)
  {
    if (std::fabs(matrix[i] - expected) > tolerance)
    {
      return false;
    }
  }
  return true;
}

/* ========================================================================== */
/* Test 1: Initialization and Configuration                                   */
/* ========================================================================== */

static void test_initialization()
{
  print_test_header("Initialization and Configuration");

  // Test 1.1: Basic initialization
  {
    int status = mil_init(nullptr);
    print_test_result("Basic initialization", status == MIL_OK,
                      status == MIL_OK ? "MIL initialized successfully" : mil_strerror(status));
  }

  // Test 1.2: Check initialization status
  {
    int is_init = mil_is_initialized();
    print_test_result("Is initialized check", is_init == 1);
  }

  // Test 1.3: Get backend
  {
    mil_backend_t backend = mil_get_backend();
    char msg[128];
    std::snprintf(msg, sizeof(msg), "Backend: %s", mil_backend_name(backend));
    print_test_result("Backend detection", backend != MIL_BACKEND_AUTO, msg);
  }

  // Test 1.4: Thread configuration
  {
    size_t num_threads = mil_get_num_threads();
    char msg[128];
    std::snprintf(msg, sizeof(msg), "Using %zu threads", num_threads);
    print_test_result("Thread count", num_threads > 0, msg);
  }

  // Test 1.5: System info
  {
    const char *info = mil_get_system_info();
    print_test_result("System info", info != nullptr);
    if (info)
    {
      std::printf("\n%s\n", info);
    }
  }
}

/* ========================================================================== */
/* Test 2: SGEMM Operations                                                   */
/* ========================================================================== */

static void test_sgemm_operations()
{
  print_test_header("SGEMM Operations");

  // Test 2.1: Small SGEMM (identity check)
  {
    size_t size = TEST_SMALL_SIZE;
    Matrix<float> A(size, size, 1.0f);
    Matrix<float> B(size, size, 1.0f);
    Matrix<float> C(size, size, 0.0f);

    if (A.valid() && B.valid() && C.valid())
    {
      mil_perf_stats_t stats = {0};
      int status = mil_sgemm(
          MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
          size, size, size,
          1.0f, A, size, B, size, 0.0f, C, size,
          &stats);

      bool correct = verify_matrix_value<float>(C, size * size, static_cast<float>(size), 0.1f);
      char msg[128];
      std::snprintf(msg, sizeof(msg), "Size %zux%zu, expected result: %.1f", size, size, static_cast<float>(size));
      print_test_result("Small SGEMM correctness", status == MIL_OK && correct, msg);

      if (status == MIL_OK)
        print_perf_stats(&stats);
    }
  } // Automatic cleanup via RAII

  // Test 2.2: Medium SGEMM (performance test)
  {
    size_t size = TEST_MEDIUM_SIZE;
    Matrix<float> A(size, size, -1.0f); // Random
    Matrix<float> B(size, size, -1.0f);
    Matrix<float> C(size, size, 0.0f);

    if (A.valid() && B.valid() && C.valid())
    {
      mil_perf_stats_t best_stats = {0};
      best_stats.gflops = 0.0;

      for (int iter = 0; iter < TEST_ITERATIONS; ++iter)
      {
        mil_perf_stats_t stats = {0};
        int status = mil_sgemm(
            MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
            size, size, size,
            1.0f, A, size, B, size, 0.0f, C, size,
            &stats);

        if (status == MIL_OK && stats.gflops > best_stats.gflops)
        {
          best_stats = stats;
        }
      }

      char msg[128];
      std::snprintf(msg, sizeof(msg), "Size %zux%zu, best: %.2f GFLOPS", size, size, best_stats.gflops);
      print_test_result("Medium SGEMM performance", best_stats.gflops > 0, msg);

      if (best_stats.gflops > 0)
        print_perf_stats(&best_stats);
    }
  }

  // Test 2.3: Rectangular GEMM
  {
    size_t m = 100, n = 200, k = 150;
    Matrix<float> A(m, k, 2.0f);
    Matrix<float> B(k, n, 3.0f);
    Matrix<float> C(m, n, 0.0f);

    if (A.valid() && B.valid() && C.valid())
    {
      mil_perf_stats_t stats = {0};
      int status = mil_sgemm(
          MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
          m, n, k,
          1.0f, A, k, B, n, 0.0f, C, n,
          &stats);

      // Expected result: each element should be 2.0 * 3.0 * k = 6.0 * k
      float expected = 6.0f * static_cast<float>(k);
      bool correct = verify_matrix_value<float>(C, m * n, expected, 1.0f);

      char msg[128];
      std::snprintf(msg, sizeof(msg), "Size %zux%zux%zu", m, n, k);
      print_test_result("Rectangular SGEMM", status == MIL_OK && correct, msg);

      if (status == MIL_OK)
        print_perf_stats(&stats);
    }
  }

  // Test 2.4: Alpha/Beta scaling
  {
    size_t size = 32;
    Matrix<float> A(size, size, 1.0f);
    Matrix<float> B(size, size, 1.0f);
    Matrix<float> C(size, size, 5.0f);

    if (A.valid() && B.valid() && C.valid())
    {
      // C = 2.0 * A * B + 3.0 * C
      // = 2.0 * size + 3.0 * 5.0 = 2*size + 15
      int status = mil_sgemm(
          MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
          size, size, size,
          2.0f, A, size, B, size, 3.0f, C, size,
          nullptr);

      float expected = 2.0f * static_cast<float>(size) + 15.0f;
      bool correct = verify_matrix_value<float>(C, size * size, expected, 0.1f);

      char msg[128];
      std::snprintf(msg, sizeof(msg), "alpha=2.0, beta=3.0, expected=%.1f", expected);
      print_test_result("Alpha/Beta scaling", status == MIL_OK && correct, msg);
    }
  }
}

/* ========================================================================== */
/* Test 3: DGEMM Operations                                                   */
/* ========================================================================== */

static void test_dgemm_operations()
{
  print_test_header("DGEMM Operations (Double Precision)");

  size_t size = TEST_SMALL_SIZE;
  Matrix<double> A(size, size, 1.0);
  Matrix<double> B(size, size, 1.0);
  Matrix<double> C(size, size, 0.0);

  if (A.valid() && B.valid() && C.valid())
  {
    mil_perf_stats_t stats = {0};
    int status = mil_dgemm(
        MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
        size, size, size,
        1.0, A, size, B, size, 0.0, C, size,
        &stats);

    bool correct = verify_matrix_value<double>(C, size * size, static_cast<double>(size), 0.01);

    char msg[128];
    std::snprintf(msg, sizeof(msg), "Size %zux%zu", size, size);
    print_test_result("DGEMM correctness", status == MIL_OK && correct, msg);

    if (status == MIL_OK)
      print_perf_stats(&stats);
  }
}

/* ========================================================================== */
/* Test 4: GEMV Operations                                                    */
/* ========================================================================== */

static void test_gemv_operations()
{
  print_test_header("GEMV Operations (Matrix-Vector Multiply)");

  size_t m = 256, n = 256;
  Matrix<float> A(m, n, 1.0f);
  Matrix<float> x(n, 1, 1.0f);
  Matrix<float> y(m, 1, 0.0f);

  if (A.valid() && x.valid() && y.valid())
  {
    mil_perf_stats_t stats = {0};
    int status = mil_sgemv(
        MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS,
        m, n,
        1.0f, A, n, x, 1, 0.0f, y, 1,
        &stats);

    bool correct = verify_matrix_value<float>(y, m, static_cast<float>(n), 0.1f);

    char msg[128];
    std::snprintf(msg, sizeof(msg), "Size %zux%zu, expected result: %.1f", m, n, static_cast<float>(n));
    print_test_result("SGEMV correctness", status == MIL_OK && correct, msg);

    if (status == MIL_OK)
      print_perf_stats(&stats);
  }
}

/* ========================================================================== */
/* Test 5: Batched Operations                                                 */
/* ========================================================================== */

static void test_batched_operations()
{
  print_test_header("Batched GEMM Operations");

  size_t size = 64;
  size_t batch_count = 8;

  // Use vector of Matrix objects for automatic cleanup
  std::vector<std::unique_ptr<Matrix<float>>> A_mats;
  std::vector<std::unique_ptr<Matrix<float>>> B_mats;
  std::vector<std::unique_ptr<Matrix<float>>> C_mats;

  std::vector<const float *> A_array(batch_count);
  std::vector<const float *> B_array(batch_count);
  std::vector<float *> C_array(batch_count);

  for (size_t b = 0; b < batch_count; ++b)
  {
    A_mats.push_back(std::make_unique<Matrix<float>>(size, size, 1.0f));
    B_mats.push_back(std::make_unique<Matrix<float>>(size, size, 1.0f));
    C_mats.push_back(std::make_unique<Matrix<float>>(size, size, 0.0f));

    A_array[b] = *A_mats[b];
    B_array[b] = *B_mats[b];
    C_array[b] = *C_mats[b];
  }

  mil_perf_stats_t stats = {0};
  int status = mil_sgemm_batch(
      MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
      size, size, size,
      1.0f, A_array.data(), size, B_array.data(), size, 0.0f, C_array.data(), size,
      batch_count,
      &stats);

  bool all_correct = true;
  for (size_t b = 0; b < batch_count; ++b)
  {
    if (!verify_matrix_value<float>(C_array[b], size * size, static_cast<float>(size), 0.1f))
    {
      all_correct = false;
      break;
    }
  }

  char msg[128];
  std::snprintf(msg, sizeof(msg), "Batch size: %zu, matrix size: %zux%zu", batch_count, size, size);
  print_test_result("Batched SGEMM", status == MIL_OK && all_correct, msg);

  if (status == MIL_OK)
    print_perf_stats(&stats);

  // Automatic cleanup via unique_ptr destructors
}

/* ========================================================================== */
/* Test 6: Convolution Operations                                             */
/* ========================================================================== */

static void test_convolution_operations()
{
  print_test_header("Convolution Operations");

  // Test 6.1: Simple 2D convolution
  {
    size_t batch = 1, in_channels = 1, out_channels = 1;
    size_t in_h = 8, in_w = 8;
    size_t kh = 3, kw = 3;
    size_t stride_h = 1, stride_w = 1;
    size_t pad_h = 0, pad_w = 0;

    Matrix<float> input(batch * in_channels * in_h * in_w, 1, 1.0f);
    Matrix<float> kernel(out_channels * in_channels * kh * kw, 1, 1.0f);

    size_t out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
    size_t out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
    Matrix<float> output(batch * out_channels * out_h * out_w, 1, 0.0f);

    if (input.valid() && kernel.valid() && output.valid())
    {
      mil_perf_stats_t stats = {0};
      int status = mil_conv2d_f32(
          input, kernel, nullptr, output,
          batch, in_channels, in_h, in_w,
          out_channels, kh, kw,
          stride_h, stride_w, pad_h, pad_w,
          &stats);

      char msg[128];
      std::snprintf(msg, sizeof(msg), "Input: %zux%zu, Kernel: %zux%zu, Output: %zux%zu",
                    in_h, in_w, kh, kw, out_h, out_w);
      print_test_result("Direct Conv2D", status == MIL_OK, msg);

      if (status == MIL_OK)
        print_perf_stats(&stats);
    }
  }

  // Test 6.2: Im2Col convolution
  {
    size_t batch = 2, in_channels = 3, out_channels = 8;
    size_t in_h = 16, in_w = 16;
    size_t kh = 3, kw = 3;

    Matrix<float> input(batch * in_channels * in_h * in_w, 1, -1.0f);
    Matrix<float> kernel(out_channels * in_channels * kh * kw, 1, -1.0f);
    Matrix<float> bias(out_channels, 1, 0.5f);

    size_t out_h = in_h - kh + 1;
    size_t out_w = in_w - kw + 1;
    Matrix<float> output(batch * out_channels * out_h * out_w, 1, 0.0f);

    if (input.valid() && kernel.valid() && output.valid())
    {
      mil_perf_stats_t stats = {0};
      int status = mil_conv2d_im2col_f32(
          input, kernel, bias, output,
          batch, in_channels, in_h, in_w,
          out_channels, kh, kw,
          1, 1, 0, 0,
          &stats);

      char msg[128];
      std::snprintf(msg, sizeof(msg), "Batch: %zu, Channels: %zu->%zu", batch, in_channels, out_channels);
      print_test_result("Im2Col Conv2D", status == MIL_OK, msg);

      if (status == MIL_OK)
        print_perf_stats(&stats);
    }
  }
}

/* ========================================================================== */
/* Test 7: Utility Functions                                                  */
/* ========================================================================== */

static void test_utility_functions()
{
  print_test_header("Utility Functions");

  // Test 7.1: Tile computation
  {
    size_t tile_m, tile_n, tile_k;
    int status = mil_compute_optimal_tiles(1024, 1024, 1024, sizeof(float),
                                           &tile_m, &tile_n, &tile_k);

    char msg[128];
    std::snprintf(msg, sizeof(msg), "Tiles: M=%zu, N=%zu, K=%zu", tile_m, tile_n, tile_k);
    print_test_result("Optimal tile computation", status == MIL_OK, msg);
  }

  // Test 7.2: Self-test
  {
    int status = mil_self_test(0); // Non-verbose
    print_test_result("Internal self-test", status == MIL_OK);
  }
}

/* ========================================================================== */
/* Test 8: Large Matrix Performance Tests                                     */
/* ========================================================================== */

static void test_large_matrix_performance()
{
  print_test_header("Large Matrix Performance Tests");

  std::printf("\nTesting large matrix multiplications to measure peak performance...\n");
  std::printf("This may take several minutes depending on your system.\n\n");

  struct TestSize
  {
    size_t size;
    const char *name;
    int iterations;
  };

  TestSize test_sizes[] = {
      {1024, "1024x1024x1024", TEST_ITERATIONS},
      {2048, "2048x2048x2048", TEST_LARGE_ITERATIONS},
      {4096, "4096x4096x4096", TEST_LARGE_ITERATIONS},
      {8192, "8192x8192x8192", 1} // Single iteration for 8192 due to size
  };

  std::printf("%-20s %-15s %-15s\n",
              "Size", "GFLOPS", "Time(ms)\n");
  print_separator('-');

  for (const auto &test : test_sizes)
  {
    size_t size = test.size;

    std::printf("Allocating %s matrices (%.2f GB)\n",
                test.name,
                (3.0 * size * size * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stdout);

    float *A = static_cast<float *>(std::malloc(size * size * sizeof(float)));
    float *B = static_cast<float *>(std::malloc(size * size * sizeof(float)));
    float *C = static_cast<float *>(std::malloc(size * size * sizeof(float)));

    if (A == nullptr || B == nullptr || C == nullptr)
    {
      std::printf("FAILED (allocation)\n");
      std::free(A);
      std::free(B);
      std::free(C);
      print_test_result("Large matrix allocation\n", false, test.name);
      continue;
    }

    std::printf("OK\n");
    std::printf("Initializing matrices\n");
    std::fflush(stdout);

    // Initialize with random values
    for (size_t i = 0; i < size * size; ++i)
    {
      A[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
      B[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
      C[i] = 0.0f;
    }

    std::printf("OK\n");
    std::printf("Running warmup\n");
    std::fflush(stdout);

    // Warmup run
    mil_sgemm(MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
              size, size, size, 1.0f, A, size, B, size, 0.0f, C, size, nullptr);

    std::printf("OK\n");
    std::printf("Running %d benchmark iteration(s)\n", test.iterations);

    // Benchmark runs
    mil_perf_stats_t best_stats = {0};
    best_stats.gflops = 0.0;

    for (int iter = 0; iter < test.iterations; ++iter)
    {
      std::printf("  Iteration %d/%d ", iter + 1, test.iterations);
      std::fflush(stdout);

      mil_perf_stats_t stats = {0};
      int status = mil_sgemm(
          MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
          size, size, size, 1.0f, A, size, B, size, 0.0f, C, size, &stats);

      if (status == MIL_OK)
      {
        std::printf("%.2f GFLOPS\n", stats.gflops);
        if (stats.gflops > best_stats.gflops)
        {
          best_stats = stats;
        }
      }
      else
      {
        std::printf("FAILED\n");
      }
    }

    // Calculate efficiency (percentage of theoretical peak)
    // Rough estimate: modern CPU ~2-4 TFLOPS theoretical peak
    double theoretical_peak = 2000.0; // 2 TFLOPS conservative estimate
    double efficiency = (best_stats.gflops / theoretical_peak) * 100.0;

    std::printf("\n%-20s %-15.2f %-15.1f %-15.2f %-15.1f%%\n",
                test.name,
                best_stats.gflops,
                best_stats.elapsed_ms,
                best_stats.bandwidth_gbps,
                efficiency);

    char msg[256];
    std::snprintf(msg, sizeof(msg), "%s: %.2f GFLOPS (%.1f%% efficiency)",
                  test.name, best_stats.gflops, efficiency);
    print_test_result("Large GEMM performance", best_stats.gflops > 0, msg);

    // Explicit cleanup
    std::free(A);
    std::free(B);
    std::free(C);

    std::printf("\n");
  }

  std::printf("\nNote: Efficiency is calculated against an estimated 2 TFLOPS theoretical peak.\n");
  std::printf("Actual theoretical peak depends on your CPU model and core count.\n\n");
}

/* ========================================================================== */
/* Test 9: Performance Benchmark                                              */
/* ========================================================================== */

static void test_performance_benchmark()
{
  print_test_header("\n\nStandard Performance Benchmark Suite");

  std::printf("\nRunning GEMM benchmark across multiple sizes...\n");
  int status = mil_benchmark_gemm(128, 1024, 128, 3);
  print_test_result("Benchmark execution", status == MIL_OK);
}

/* ========================================================================== */
/* Main Test Runner                                                            */
/* ========================================================================== */

int main(int argc, char **argv)
{
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  std::printf("\n");
  print_separator('*');
  std::printf("%s", ANSI_COLOR_MAGENTA);
  std::printf("  MICROKERNEL INTERFACE LAYER (MIL) - COMPREHENSIVE TEST SUITE\n");
  std::printf("  Project JCore - Derived Component (C++ Implementation)\n");
  std::printf("%s", ANSI_COLOR_RESET);
  print_separator('*');
  std::printf("\n");

  // Run all tests
  test_initialization();
  test_sgemm_operations();
  test_dgemm_operations();
  test_gemv_operations();
  test_batched_operations();
  test_convolution_operations();
  test_utility_functions();
  test_performance_benchmark();
  test_large_matrix_performance(); // New comprehensive large matrix tests

  // Final summary
  std::printf("\n");
  print_separator('*');
  std::printf("%sFINAL TEST SUMMARY%s\n", ANSI_COLOR_CYAN, ANSI_COLOR_RESET);
  print_separator('*');

  std::printf("\n");
  std::printf("Total Tests:    %d\n", g_test_stats.total_tests);
  std::printf("%sPassed:         %d%s\n", ANSI_COLOR_GREEN, g_test_stats.passed_tests, ANSI_COLOR_RESET);
  std::printf("%sFailed:         %d%s\n", ANSI_COLOR_RED, g_test_stats.failed_tests, ANSI_COLOR_RESET);
  std::printf("Skipped:        %d\n", g_test_stats.skipped_tests);

  std::printf("\n");
  std::printf("Total Time:     %.2f ms\n", g_test_stats.total_time_ms);

  double success_rate = static_cast<double>(g_test_stats.passed_tests) /
                        g_test_stats.total_tests * 100.0;
  std::printf("\nSuccess Rate:   %.1f%%\n", success_rate);

  std::printf("\n");
  print_separator('*');

  if (g_test_stats.failed_tests == 0)
  {
    std::printf("%s\n✓ ALL TESTS PASSED!%s\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
  }
  else
  {
    std::printf("%s\n✗ SOME TESTS FAILED%s\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);
  }

  std::printf("\n");

  // Cleanup
  mil_shutdown();

  return (g_test_stats.failed_tests == 0) ? 0 : 1;
}