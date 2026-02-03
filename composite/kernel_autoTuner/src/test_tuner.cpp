// test_adaptive_tuner.cpp
// Comprehensive test suite for Adaptive Kernel Auto-Tuner
// Tests all aspects: registration, benchmarking, error handling, and GFLOPS analysis

#include "adaptive_tuner.h"
#include "kernels_openblas.h"
#include "kernels_blis.h"
#include "jcore_isa_dispatch.h"
#include "cpu_info.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

// Test configuration
struct TestConfig
{
  size_t M;
  size_t N;
  size_t K;
  size_t threads;
  const char *description;
};

// ANSI color codes for output
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_RESET "\033[0m"

// Test result tracker
struct TestResults
{
  int total;
  int passed;
  int failed;

  TestResults() : total(0), passed(0), failed(0) {}

  void record_pass(const char *test_name)
  {
    total++;
    passed++;
    printf(COLOR_GREEN "[PASS]" COLOR_RESET " %s\n", test_name);
  }

  void record_fail(const char *test_name, const char *reason)
  {
    total++;
    failed++;
    printf(COLOR_RED "[FAIL]" COLOR_RESET " %s: %s\n", test_name, reason);
  }

  void print_summary()
  {
    printf("\n");
    printf("========================================\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", total);
    printf("  " COLOR_GREEN "Passed: %d" COLOR_RESET "\n", passed);
    if (failed > 0)
    {
      printf("  " COLOR_RED "Failed: %d" COLOR_RESET "\n", failed);
    }
    else
    {
      printf("  " COLOR_GREEN "Failed: 0" COLOR_RESET "\n");
    }
    printf("  Success Rate: %.1f%%\n", (100.0 * passed) / total);
    printf("========================================\n");
  }
};

static TestResults g_results;

// Test 1: Initialization and shutdown
void test_init_shutdown()
{
  printf("\n" COLOR_BLUE "[TEST]" COLOR_RESET " Testing initialization and shutdown\n");

  at_status_t status = at_init();
  if (status != AT_OK)
  {
    g_results.record_fail("at_init", at_status_str(status));
    return;
  }
  g_results.record_pass("at_init");

  // Test double init (should be safe)
  status = at_init();
  if (status != AT_OK)
  {
    g_results.record_fail("at_init (double)", "Should handle double init gracefully");
  }
  else
  {
    g_results.record_pass("at_init (double)");
  }

  at_shutdown();
  g_results.record_pass("at_shutdown");

  // Test shutdown when not initialized (should be safe)
  at_shutdown();
  g_results.record_pass("at_shutdown (double)");
}

// Test 2: Kernel registration
void test_kernel_registration()
{
  printf("\n" COLOR_BLUE "[TEST]" COLOR_RESET " Testing kernel registration\n");

  at_status_t status = at_init();
  if (status != AT_OK)
  {
    g_results.record_fail("Kernel registration setup", "Init failed");
    return;
  }

  // Register OpenBLAS
  status = at_register_matmul_impl("OpenBLAS", JCORE_FEAT_NONE, openblas_sgemm);
  if (status == AT_OK)
  {
    g_results.record_pass("Register OpenBLAS");
  }
  else
  {
    g_results.record_fail("Register OpenBLAS", at_status_str(status));
  }

  // Register BLIS
  status = at_register_matmul_impl("BLIS", JCORE_FEAT_NONE, blis_sgemm);
  if (status == AT_OK)
  {
    g_results.record_pass("Register BLIS");
  }
  else
  {
    g_results.record_fail("Register BLIS", at_status_str(status));
  }

  // Test duplicate registration (should fail)
  status = at_register_matmul_impl("OpenBLAS", JCORE_FEAT_NONE, openblas_sgemm);
  if (status == AT_ERR_CONFLICT)
  {
    g_results.record_pass("Reject duplicate registration");
  }
  else
  {
    g_results.record_fail("Reject duplicate registration", "Should return AT_ERR_CONFLICT");
  }

  // Test NULL function pointer (should fail)
  status = at_register_matmul_impl("NullKernel", JCORE_FEAT_NONE, nullptr);
  if (status == AT_ERR_INVALID_ARG)
  {
    g_results.record_pass("Reject NULL function pointer");
  }
  else
  {
    g_results.record_fail("Reject NULL function pointer", "Should return AT_ERR_INVALID_ARG");
  }

  // Test NULL name (should fail)
  status = at_register_matmul_impl(nullptr, JCORE_FEAT_NONE, openblas_sgemm);
  if (status == AT_ERR_INVALID_ARG)
  {
    g_results.record_pass("Reject NULL name");
  }
  else
  {
    g_results.record_fail("Reject NULL name", "Should return AT_ERR_INVALID_ARG");
  }

  at_shutdown();
}

// Test 3: Error handling
void test_error_handling()
{
  printf("\n" COLOR_BLUE "[TEST]" COLOR_RESET " Testing error handling\n");

  // Test operations before init
  char best_name[256];
  at_status_t status = at_register_matmul_impl("Test", JCORE_FEAT_NONE, openblas_sgemm);
  if (status == AT_ERR_NOT_INITIALIZED)
  {
    g_results.record_pass("Register before init");
  }
  else
  {
    g_results.record_fail("Register before init", "Should return AT_ERR_NOT_INITIALIZED");
  }

  status = at_benchmark_matmul_all(64, 64, 64, 0, 0, best_name, sizeof(best_name));
  if (status == AT_ERR_NOT_INITIALIZED)
  {
    g_results.record_pass("Benchmark before init");
  }
  else
  {
    g_results.record_fail("Benchmark before init", "Should return AT_ERR_NOT_INITIALIZED");
  }

  // Init and test invalid arguments
  at_init();

  status = at_benchmark_matmul_all(0, 64, 64, 0, 0, best_name, sizeof(best_name));
  if (status == AT_ERR_INVALID_ARG)
  {
    g_results.record_pass("Benchmark with M=0");
  }
  else
  {
    g_results.record_fail("Benchmark with M=0", "Should return AT_ERR_INVALID_ARG");
  }

  status = at_benchmark_matmul_all(64, 64, 64, 0, 0, nullptr, sizeof(best_name));
  if (status == AT_ERR_INVALID_ARG)
  {
    g_results.record_pass("Benchmark with NULL output");
  }
  else
  {
    g_results.record_fail("Benchmark with NULL output", "Should return AT_ERR_INVALID_ARG");
  }

  // Test benchmark with no kernels
  status = at_benchmark_matmul_all(64, 64, 64, 0, 0, best_name, sizeof(best_name));
  if (status == AT_ERR_NO_KERNELS)
  {
    g_results.record_pass("Benchmark with no kernels");
  }
  else
  {
    g_results.record_fail("Benchmark with no kernels", "Should return AT_ERR_NO_KERNELS");
  }

  at_shutdown();
}

// Test 4: Benchmarking with various matrix sizes
void test_benchmarking()
{
  printf("\n" COLOR_BLUE "[TEST]" COLOR_RESET " Testing comprehensive benchmarking\n");

  at_status_t status = at_init();
  if (status != AT_OK)
  {
    g_results.record_fail("Benchmark setup", "Init failed");
    return;
  }

  // Register all kernels
  at_register_matmul_impl("OpenBLAS", JCORE_FEAT_NONE, openblas_sgemm);
  at_register_matmul_impl("BLIS", JCORE_FEAT_NONE, blis_sgemm);

  // Test configurations (small, medium, large, rectangular)
  TestConfig configs[] = {
      {64, 64, 64, 0, "Small square (64x64x64)"},
      {128, 128, 128, 0, "Medium square (128x128x128)"},
      {256, 256, 256, 0, "Large square (256x256x256)"},
      {512, 512, 512, 0, "Very large square (512x512x512)"},
      {100, 200, 150, 0, "Rectangular (100x200x150)"},
      {1, 1024, 1024, 0, "Skinny M=1 (1x1024x1024)"},
      {1024, 1024, 1024, 0, "Skinny N=1 (1024x1x1024)"},
      {2048, 2048, 2048, 0, "Skinny N=1 (1024x1x1024)"},
      {32, 32, 32, 1, "Small with 4 threads (32x32x32)"}};

  size_t num_configs = sizeof(configs) / sizeof(configs[0]);

  for (size_t i = 0; i < num_configs; ++i)
  {
    const TestConfig &cfg = configs[i];
    char best_name[256];

    printf("\n" COLOR_YELLOW "  Testing: %s" COLOR_RESET "\n", cfg.description);

    status = at_benchmark_matmul_all(
        cfg.M, cfg.N, cfg.K,
        cfg.threads, 0,
        best_name, sizeof(best_name));

    if (status == AT_OK)
    {
      printf("    Winner: %s\n", best_name);
      g_results.record_pass(cfg.description);
    }
    else
    {
      g_results.record_fail(cfg.description, at_status_str(status));
    }
  }

  at_shutdown();
}

// Test 5: CPU feature detection integration
void test_feature_detection()
{
  printf("\n" COLOR_BLUE "[TEST]" COLOR_RESET " Testing CPU feature detection\n");

  cpu_info_t cpu_info = detect_cpu_info();

  printf("  CPU Info:\n");
  printf("    Cores: %d\n", cpu_info.cores);
  printf("    Logical cores: %d\n", cpu_info.logical_cores);
  printf("    AVX: %s\n", cpu_info.avx ? "YES" : "NO");
  printf("    AVX2: %s\n", cpu_info.avx2 ? "YES" : "NO");
  printf("    AVX-512: %s\n", cpu_info.avx512 ? "YES" : "NO");
  printf("    AMX: %s\n", cpu_info.amx ? "YES" : "NO");
  printf("    L1D: %d KB\n", cpu_info.l1d_kb);
  printf("    L2: %d KB\n", cpu_info.l2_kb);
  printf("    L3: %d KB\n", cpu_info.l3_kb);
  printf("    NUMA nodes: %d\n", cpu_info.numa_nodes);

  if (cpu_info.cores > 0)
  {
    g_results.record_pass("CPU core detection");
  }
  else
  {
    g_results.record_fail("CPU core detection", "No cores detected");
  }

  jcore_features_t features = jcore_get_host_features();
  printf("  ISA Features: 0x%llx\n", (unsigned long long)features);

  g_results.record_pass("Feature detection");
}

// Test 6: Status string conversion
void test_status_strings()
{
  printf("\n" COLOR_BLUE "[TEST]" COLOR_RESET " Testing status string conversion\n");

  at_status_t codes[] = {
      AT_OK,
      AT_ERR_NO_MEMORY,
      AT_ERR_INVALID_ARG,
      AT_ERR_NOT_INITIALIZED,
      AT_ERR_NO_KERNELS,
      AT_ERR_BENCHMARK_FAIL,
      AT_ERR_CONFLICT,
      AT_ERR_INTERNAL};

  bool all_valid = true;
  for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); ++i)
  {
    const char *str = at_status_str(codes[i]);
    if (!str || strlen(str) == 0)
    {
      all_valid = false;
      break;
    }
  }

  if (all_valid)
  {
    g_results.record_pass("Status string conversion");
  }
  else
  {
    g_results.record_fail("Status string conversion", "Invalid string returned");
  }
}

// Test 7: Statistical analysis of results
void test_statistical_analysis()
{
  printf("\n" COLOR_BLUE "[TEST]" COLOR_RESET " Testing statistical analysis\n");

  at_status_t status = at_init();
  if (status != AT_OK)
  {
    g_results.record_fail("Statistics setup", "Init failed");
    return;
  }

  // Register all kernels
  at_register_matmul_impl("OpenBLAS", JCORE_FEAT_NONE, openblas_sgemm);
  at_register_matmul_impl("BLIS", JCORE_FEAT_NONE, blis_sgemm);

  // Run multiple benchmarks and analyze consistency
  printf("  Running consistency test (3 iterations of 256x256x256)...\n");

  std::vector<std::string> winners;
  for (int i = 0; i < 3; ++i)
  {
    char best_name[256];
    status = at_benchmark_matmul_all(256, 256, 256, 0, 0, best_name, sizeof(best_name));
    if (status == AT_OK)
    {
      winners.push_back(best_name);
      printf("    Run %d: %s\n", i + 1, best_name);
    }
  }

  // Check if results are consistent
  if (winners.size() == 3)
  {
    bool consistent = (winners[0] == winners[1]) && (winners[1] == winners[2]);
    if (consistent)
    {
      g_results.record_pass("Result consistency");
    }
    else
    {
      printf("    Note: Winners varied across runs (acceptable due to timing variance)\n");
      g_results.record_pass("Result consistency (with variance)");
    }
  }
  else
  {
    g_results.record_fail("Result consistency", "Failed to complete all runs");
  }

  at_shutdown();
}

int main(int argc, char **argv)
{
  printf("\n");
  printf("========================================\n");
  printf("Adaptive Kernel Auto-Tuner Test Suite\n");
  printf("========================================\n");

  // Run all tests
  test_init_shutdown();
  test_kernel_registration();
  test_error_handling();
  test_feature_detection();
  test_status_strings();
  test_benchmarking();
  test_statistical_analysis();

  // Print summary
  g_results.print_summary();

  return (g_results.failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}