// composite/tuning_cache/src/test_cached_autotuner.cpp
//
// Test Suite for Cached AutoTuner
//
// SCOPE: Tests the cache layer functionality, NOT the tuning quality.
//        This validates that the cache correctly:
//        - Stores tuning decisions
//        - Retrieves cached results
//        - Persists and loads from disk
//        - Handles edge cases (clear, export, import)
//
// NOTE: The AutoTuner may return score=0 for all kernels if actual
//       benchmarking is not implemented. This is expected - the cache
//       layer stores whatever decisions the tuner makes.
//
// PERFORMANCE: Speedup metrics (1000x+) measure cache lookup speed
//              vs. calling the AutoTuner, NOT end-to-end GEMM performance.

#include "cached_autotuner.h"
#include <cstdio>
#include <cstring>
#include <ctime>

#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_CYAN "\033[36m"
#define COLOR_BOLD "\033[1m"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_START(name)                                   \
  printf(COLOR_BLUE "[TEST] " COLOR_RESET "%s... ", name); \
  fflush(stdout)

#define TEST_PASS(name)                          \
  do                                             \
  {                                              \
    printf(COLOR_GREEN "PASS" COLOR_RESET "\n"); \
    tests_passed++;                              \
  } while (0)

#define TEST_FAIL(name, msg)                             \
  do                                                     \
  {                                                      \
    printf(COLOR_RED "FAIL" COLOR_RESET " - %s\n", msg); \
    tests_failed++;                                      \
  } while (0)

#define ASSERT(cond, msg) \
  do                      \
  {                       \
    if (!(cond))          \
    {                     \
      TEST_FAIL("", msg); \
      return;             \
    }                     \
  } while (0)

void test_init_shutdown()
{
  TEST_START("Init and Shutdown");

  cat_handle_t *handle = nullptr;
  cat_status_t s = cat_init(&handle);

  ASSERT(s == CAT_OK, "Init failed");
  ASSERT(handle != nullptr, "Handle is NULL");

  cat_shutdown(handle);

  TEST_PASS("");
}

void test_config()
{
  TEST_START("Configuration");

  cat_config_t config;
  cat_config_init_default(&config);

  ASSERT(strlen(config.cache_dir) > 0, "Cache dir not set");
  ASSERT(config.format == CAT_FORMAT_BINARY, "Format should be binary");
  ASSERT(config.validate_hardware == 1, "Validate HW should be enabled");

  TEST_PASS("");
}

void test_select_kernel_first_run()
{
  TEST_START("Select Kernel (First Run - Cache Miss)");

  cat_handle_t *handle = nullptr;
  cat_status_t s = cat_init(&handle);
  ASSERT(s == CAT_OK, "Init failed");

  // Clear cache for clean test
  cat_clear_cache(handle);

  const char *kernel = cat_select_kernel(handle, 512, 512, 512, 4, 64);
  ASSERT(kernel != nullptr, "Kernel selection failed");
  ASSERT(strlen(kernel) > 0, "Kernel name is empty");

  printf("\n  Selected: %s ", kernel);

  cat_stats_t stats;
  cat_get_stats(handle, &stats);
  ASSERT(stats.cache_misses == 1, "Should have 1 cache miss");
  ASSERT(stats.benchmarks_run == 1, "Should have 1 benchmark");

  cat_shutdown(handle);
  TEST_PASS("");
}

void test_select_kernel_second_run()
{
  TEST_START("Select Kernel (Second Run - Cache Hit)");

  cat_handle_t *handle = nullptr;
  cat_status_t s = cat_init(&handle);
  ASSERT(s == CAT_OK, "Init failed");

  // First call - populates cache
  const char *kernel1 = cat_select_kernel(handle, 1024, 1024, 1024, 8, 128);
  ASSERT(kernel1 != NULL, "First kernel selection failed");

  // Second call - should hit cache
  const char *kernel2 = cat_select_kernel(handle, 1024, 1024, 1024, 8, 128);
  ASSERT(kernel2 != NULL, "Second kernel selection failed");
  ASSERT(strcmp(kernel1, kernel2) == 0, "Kernels should match");

  cat_stats_t stats;
  cat_get_stats(handle, &stats);
  ASSERT(stats.cache_hits == 1, "Should have 1 cache hit");

  printf("\n  Kernel: %s, Hit Rate: %.1f%% ", kernel2, stats.hit_rate * 100.0);

  cat_shutdown(handle);
  TEST_PASS("");
}

void test_multiple_workloads()
{
  TEST_START("Multiple Workloads");

  cat_handle_t *handle = nullptr;
  cat_status_t s = cat_init(&handle);
  ASSERT(s == CAT_OK, "Init failed");

  cat_clear_cache(handle);

  // Different workloads
  struct
  {
    size_t M, N, K, threads, tile;
  } workloads[] = {
      {512, 512, 512, 4, 64},
      {1024, 1024, 1024, 8, 128},
      {2048, 2048, 2048, 16, 256},
      {512, 512, 512, 4, 64}, // Duplicate - should hit cache
  };

  for (size_t i = 0; i < sizeof(workloads) / sizeof(workloads[0]); i++)
  {
    const char *kernel = cat_select_kernel(handle,
                                           workloads[i].M, workloads[i].N, workloads[i].K,
                                           workloads[i].threads, workloads[i].tile);
    ASSERT(kernel != NULL, "Kernel selection failed");
  }

  cat_stats_t stats;
  cat_get_stats(handle, &stats);
  ASSERT(stats.total_entries == 3, "Should have 3 unique entries");
  ASSERT(stats.cache_hits == 1, "Should have 1 hit");
  ASSERT(stats.cache_misses == 3, "Should have 3 misses");

  printf("\n  Entries: %zu, Hits: %zu, Misses: %zu ",
         stats.total_entries, stats.cache_hits, stats.cache_misses);

  cat_shutdown(handle);
  TEST_PASS("");
}

void test_save_load()
{
  TEST_START("Save and Load Cache");

  // First handle - populate cache
  cat_handle_t *handle1 = nullptr;
  cat_status_t s = cat_init(&handle1);
  ASSERT(s == CAT_OK, "Init failed");

  cat_clear_cache(handle1);

  cat_select_kernel(handle1, 512, 512, 512, 4, 64);
  cat_select_kernel(handle1, 1024, 1024, 1024, 8, 128);

  s = cat_save_cache(handle1);
  ASSERT(s == CAT_OK, "Save failed");

  cat_shutdown(handle1);

  // Second handle - load cache
  cat_handle_t *handle2 = nullptr;
  s = cat_init(&handle2);
  ASSERT(s == CAT_OK, "Init failed");

  s = cat_load_cache(handle2);
  ASSERT(s == CAT_OK, "Load failed");

  // Should hit cache now
  const char *kernel = cat_select_kernel(handle2, 512, 512, 512, 4, 64);
  ASSERT(kernel != NULL, "Kernel selection failed");

  cat_stats_t stats;
  cat_get_stats(handle2, &stats);
  ASSERT(stats.cache_hits == 1, "Should have cache hit after load");

  printf("\n  Loaded cache, hit: %s ", kernel);

  cat_shutdown(handle2);
  TEST_PASS("");
}

void test_force_benchmark()
{
  TEST_START("Force Benchmark");

  cat_handle_t *handle = nullptr;
  cat_status_t s = cat_init(&handle);
  ASSERT(s == CAT_OK, "Init failed");

  char kernel[256];
  s = cat_force_benchmark(handle, 512, 512, 512, 4, 64, kernel, sizeof(kernel));
  ASSERT(s == CAT_OK, "Force benchmark failed");
  ASSERT(strlen(kernel) > 0, "Kernel name empty");

  cat_stats_t stats;
  cat_get_stats(handle, &stats);
  ASSERT(stats.benchmarks_run == 1, "Should have 1 benchmark");

  printf("\n  Forced: %s ", kernel);

  cat_shutdown(handle);
  TEST_PASS("");
}

void test_clear_cache()
{
  TEST_START("Clear Cache");

  cat_handle_t *handle = nullptr;
  cat_status_t s = cat_init(&handle);
  ASSERT(s == CAT_OK, "Init failed");

  // Clear cache first for clean test
  cat_clear_cache(handle);

  // Add entries
  const char *k1 = cat_select_kernel(handle, 512, 512, 512, 4, 64);
  ASSERT(k1 != NULL, "First kernel selection failed");

  const char *k2 = cat_select_kernel(handle, 1024, 1024, 1024, 8, 128);
  ASSERT(k2 != NULL, "Second kernel selection failed");

  cat_stats_t stats;
  cat_get_stats(handle, &stats);
  ASSERT(stats.total_entries >= 2, "Should have at least 2 entries");

  size_t entries_before = stats.total_entries;

  s = cat_clear_cache(handle);
  ASSERT(s == CAT_OK, "Clear failed");

  cat_get_stats(handle, &stats);
  ASSERT(stats.total_entries == 0, "Should have 0 entries after clear");

  printf("\n  Cleared %zu entries ", entries_before);

  cat_shutdown(handle);
  TEST_PASS("");
}

void test_export_import()
{
  TEST_START("Export and Import Cache");

  // Export
  cat_handle_t *handle1 = nullptr;
  cat_status_t s = cat_init(&handle1);
  ASSERT(s == CAT_OK, "Init failed");

  cat_clear_cache(handle1);
  cat_select_kernel(handle1, 512, 512, 512, 4, 64);

  const char *export_path = "/tmp/jcore_test_export.bin";
  s = cat_export_cache(handle1, export_path, CAT_FORMAT_BINARY);
  ASSERT(s == CAT_OK, "Export failed");

  cat_shutdown(handle1);

  // Import
  cat_handle_t *handle2 = nullptr;
  s = cat_init(&handle2);
  ASSERT(s == CAT_OK, "Init failed");

  cat_clear_cache(handle2);

  s = cat_import_cache(handle2, export_path);
  ASSERT(s == CAT_OK, "Import failed");

  const char *kernel = cat_select_kernel(handle2, 512, 512, 512, 4, 64);
  ASSERT(kernel != NULL, "Kernel selection failed");

  cat_stats_t stats;
  cat_get_stats(handle2, &stats);
  ASSERT(stats.cache_hits == 1, "Should have cache hit after import");

  printf("\n  Imported: %s ", kernel);

  cat_shutdown(handle2);
  TEST_PASS("");
}

void test_performance()
{
  TEST_START("Performance Benchmark");

  cat_handle_t *handle = nullptr;
  cat_status_t s = cat_init(&handle);
  ASSERT(s == CAT_OK, "Init failed");

  cat_clear_cache(handle);

  // Populate cache with various workloads
  const int num_workloads = 20;
  clock_t start = clock();

  for (int i = 0; i < num_workloads; i++)
  {
    size_t M = 512 + (i * 64);
    cat_select_kernel(handle, M, M, M, 4, 64);
  }

  clock_t populate_time = clock() - start;

  // Query from cache (should be fast)
  start = clock();
  for (int i = 0; i < num_workloads; i++)
  {
    size_t M = 512 + (i * 64);
    cat_select_kernel(handle, M, M, M, 4, 64);
  }
  clock_t query_time = clock() - start;

  cat_stats_t stats;
  cat_get_stats(handle, &stats);

  double populate_ms = static_cast<double>(populate_time) / CLOCKS_PER_SEC * 1000.0;
  double query_ms = static_cast<double>(query_time) / CLOCKS_PER_SEC * 1000.0;

  printf("\n  Populate: %.2f ms, Query: %.2f ms ", populate_ms, query_ms);
  printf("\n  Speedup: %.1fx ", populate_ms / (query_ms + 0.001));
  printf("\n  Hit Rate: %.1f%% ", stats.hit_rate * 100.0);

  cat_shutdown(handle);
  TEST_PASS("");
}

int main()
{
  printf("\n");
  printf(COLOR_BOLD COLOR_CYAN);
  printf("╔════════════════════════════════════════╗\n");
  printf("║  Cached AutoTuner Test Suite           ║\n");
  printf("║  Tuning Result Cache System            ║\n");
  printf("╚════════════════════════════════════════╝\n");
  printf(COLOR_RESET);
  printf("\n");

  test_init_shutdown();
  test_config();
  test_select_kernel_first_run();
  test_select_kernel_second_run();
  test_multiple_workloads();
  test_save_load();
  test_force_benchmark();
  test_clear_cache();
  test_export_import();
  test_performance();

  printf("\n");
  printf(COLOR_BOLD);
  printf("========================================\n");
  printf("       TEST SUMMARY\n");
  printf("========================================\n");
  printf(COLOR_RESET);
  printf("Total Tests:   %d\n", tests_passed + tests_failed);
  printf(COLOR_GREEN "Passed:        %d\n" COLOR_RESET, tests_passed);
  if (tests_failed > 0)
    printf(COLOR_RED "Failed:        %d\n" COLOR_RESET, tests_failed);
  else
    printf("Failed:        %d\n", tests_failed);
  printf(COLOR_BOLD);
  printf("========================================\n");
  printf(COLOR_RESET);

  if (tests_failed == 0)
  {
    printf(COLOR_GREEN COLOR_BOLD "ALL TESTS PASSED! ✓\n" COLOR_RESET);
  }
  else
  {
    printf(COLOR_RED COLOR_BOLD "SOME TESTS FAILED! ✗\n" COLOR_RESET);
  }

  printf("\n");

  return (tests_failed == 0) ? 0 : 1;
}