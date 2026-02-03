#include "microkernel_interface.h"
#include "ffm_cache_block.h"
#include "cpu_info.h"

#include "pool_manager.h"
#include <mutex>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>

/* External references */
extern "C"
{
  extern int mil_is_initialized();
  extern mil_backend_t mil_get_backend();
}

namespace {
  pm_t *g_utils_pool = nullptr;
  std::mutex g_utils_pool_mutex;

  pm_t* GetUtilsPool() {
    std::lock_guard<std::mutex> lock(g_utils_pool_mutex);
    if (!g_utils_pool) {
      size_t pool_size = 64 * 1024 * 1024; // 64 MB
      size_t chunk_size = 128 * 1024; // 128 KB chunks for matrix allocations
      pm_status_t status = pm_init(&g_utils_pool, pool_size, chunk_size, 0, -1);
      if (status != PM_OK) {
        g_utils_pool = nullptr;
      }
    }
    return g_utils_pool;
  }

  void CleanupUtilsPool() {
    std::lock_guard<std::mutex> lock(g_utils_pool_mutex);
    if (g_utils_pool) {
      pm_shutdown(g_utils_pool);
      g_utils_pool = nullptr;
    }
  }
}

/* ========================================================================== */
/* Compute Optimal Tile Sizes                                                 */
/* ========================================================================== */

extern "C"
{

  int mil_compute_optimal_tiles(
      size_t m, size_t n, size_t k,
      size_t elem_size,
      size_t *tile_m, size_t *tile_n, size_t *tile_k)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (tile_m == nullptr || tile_n == nullptr || tile_k == nullptr)
    {
      return MIL_ERR_INVALID_ARG;
    }

    // Get CPU info
    cpu_info_t cpu_info = detect_cpu_info();

    // Initialize cache info if needed
    ffm_cache_info_t *cache_info = ffm_cache_init();
    if (cache_info == nullptr)
    {
      // Fallback to conservative defaults
      *tile_m = 64;
      *tile_n = 64;
      *tile_k = 64;
      return MIL_OK;
    }

    // Compute tile for L2 cache (typically best for matrix multiply)
    // Use 75% occupancy factor to leave room for other data
    double occupancy = 0.75;

    // Compute base tile dimension from L2 cache
    size_t l2_tile = ffm_cache_compute_tile(cache_info, 2, elem_size, occupancy);

    if (l2_tile == 0)
    {
      // Fallback if cache detection failed
      l2_tile = 128;
    }

    // For GEMM, we want tiles such that:
    // - tile_m * tile_k (panel of A) fits in L2
    // - tile_k * tile_n (panel of B) fits in L2
    // - tile_m * tile_n (block of C) fits in L1

    // Start with square tiles based on L2
    *tile_m = l2_tile;
    *tile_n = l2_tile;
    *tile_k = l2_tile;

    // Adjust based on problem size
    if (m < l2_tile)
      *tile_m = m;
    if (n < l2_tile)
      *tile_n = n;
    if (k < l2_tile)
      *tile_k = k;

    // Ensure tiles are at least 16 (minimum efficient size)
    *tile_m = std::max(size_t{16}, *tile_m);
    *tile_n = std::max(size_t{16}, *tile_n);
    *tile_k = std::max(size_t{16}, *tile_k);

    // Ensure tiles don't exceed problem dimensions
    *tile_m = std::min(m, *tile_m);
    *tile_n = std::min(n, *tile_n);
    *tile_k = std::min(k, *tile_k);

    ffm_cache_free(cache_info);

    return MIL_OK;
  }

  /* ========================================================================== */
  /* Self-Test Implementation                                                    */
  /* ========================================================================== */

  int mil_self_test(int verbose)
  {
    if (!mil_is_initialized())
    {
      std::fprintf(stderr, "[MIL Self-Test] ERROR: MIL not initialized\n");
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (verbose)
    {
      std::printf("\n========================================\n");
      std::printf("MIL Self-Test Suite\n");
      std::printf("========================================\n\n");
    }

    int tests_passed = 0;
    int tests_failed = 0;

    // Test 1: Small SGEMM
    {
      if (verbose)
        std::printf("Test 1: SGEMM (32x32x32)... ");

      size_t size = 32;
      pm_t *pool = GetUtilsPool();
      float *A = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(size * size * sizeof(float)));
      float *B = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(size * size * sizeof(float)));
      float *C = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(size * size * sizeof(float)));

      if (A == nullptr || B == nullptr || C == nullptr)
      {
        if (verbose)
          std::printf("FAILED (allocation)\n");
        tests_failed++;
      }
      else
      {
        // Initialize with simple values
        for (size_t i = 0; i < size * size; ++i)
        {
          A[i] = 1.0f;
          B[i] = 1.0f;
          C[i] = 0.0f;
        }

        mil_perf_stats_t stats = {0};
        int status = mil_sgemm(
            MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
            size, size, size,
            1.0f, A, size, B, size,
            0.0f, C, size,
            &stats);

        if (status == MIL_OK)
        {
          // Check result: each element should be approximately 'size'
          int correct = 1;
          for (size_t i = 0; i < size * size; ++i)
          {
            if (std::fabs(C[i] - static_cast<float>(size)) > 0.01f)
            {
              correct = 0;
              break;
            }
          }

          if (correct)
          {
            if (verbose)
              std::printf("PASSED (%.2f GFLOPS)\n", stats.gflops);
            tests_passed++;
          }
          else
          {
            if (verbose)
              std::printf("FAILED (incorrect result)\n");
            tests_failed++;
          }
        }
        else
        {
          if (verbose)
            std::printf("FAILED (error code: %d)\n", status);
          tests_failed++;
        }

        if (pool) {
          pm_free(pool, A);
          pm_free(pool, B);
          pm_free(pool, C);
        } else {
          std::free(A);
          std::free(B);
          std::free(C);
        }
      }
    }

    // Test 2: DGEMM
    {
      if (verbose)
        std::printf("Test 2: DGEMM (32x32x32)... ");

      size_t size = 32;
      pm_t *pool = GetUtilsPool();
      double *A = pool ? static_cast<double *>(pm_alloc(pool)) : static_cast<double *>(std::malloc(size * size * sizeof(double)));
      double *B = pool ? static_cast<double *>(pm_alloc(pool)) : static_cast<double *>(std::malloc(size * size * sizeof(double)));
      double *C = pool ? static_cast<double *>(pm_alloc(pool)) : static_cast<double *>(std::malloc(size * size * sizeof(double)));

      if (A == nullptr || B == nullptr || C == nullptr)
      {
        if (verbose)
          std::printf("FAILED (allocation)\n");
        tests_failed++;
      }
      else
      {
        for (size_t i = 0; i < size * size; ++i)
        {
          A[i] = 1.0;
          B[i] = 1.0;
          C[i] = 0.0;
        }

        mil_perf_stats_t stats = {0};
        int status = mil_dgemm(
            MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
            size, size, size,
            1.0, A, size, B, size,
            0.0, C, size,
            &stats);

        if (status == MIL_OK)
        {
          int correct = 1;
          for (size_t i = 0; i < size * size; ++i)
          {
            if (std::fabs(C[i] - static_cast<double>(size)) > 0.01)
            {
              correct = 0;
              break;
            }
          }

          if (correct)
          {
            if (verbose)
              std::printf("PASSED (%.2f GFLOPS)\n", stats.gflops);
            tests_passed++;
          }
          else
          {
            if (verbose)
              std::printf("FAILED (incorrect result)\n");
            tests_failed++;
          }
        }
        else
        {
          if (verbose)
            std::printf("FAILED (error code: %d)\n", status);
          tests_failed++;
        }

        if (pool) {
          pm_free(pool, A);
          pm_free(pool, B);
          pm_free(pool, C);
        } else {
          std::free(A);
          std::free(B);
          std::free(C);
        }
      }
    }

    // Test 3: SGEMV
    {
      if (verbose)
        std::printf("Test 3: SGEMV (128x128)... ");

      size_t m = 128, n = 128;
      pm_t *pool = GetUtilsPool();
      float *A = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(m * n * sizeof(float)));
      float *x = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(n * sizeof(float)));
      float *y = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(m * sizeof(float)));

      if (A == nullptr || x == nullptr || y == nullptr)
      {
        if (verbose)
          std::printf("FAILED (allocation)\n");
        tests_failed++;
      }
      else
      {
        for (size_t i = 0; i < m * n; ++i)
          A[i] = 1.0f;
        for (size_t i = 0; i < n; ++i)
          x[i] = 1.0f;
        for (size_t i = 0; i < m; ++i)
          y[i] = 0.0f;

        int status = mil_sgemv(
            MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS,
            m, n, 1.0f, A, n, x, 1, 0.0f, y, 1,
            nullptr);

        if (status == MIL_OK)
        {
          int correct = 1;
          for (size_t i = 0; i < m; ++i)
          {
            if (std::fabs(y[i] - static_cast<float>(n)) > 0.01f)
            {
              correct = 0;
              break;
            }
          }

          if (correct)
          {
            if (verbose)
              std::printf("PASSED\n");
            tests_passed++;
          }
          else
          {
            if (verbose)
              std::printf("FAILED (incorrect result)\n");
            tests_failed++;
          }
        }
        else
        {
          if (verbose)
            std::printf("FAILED (error code: %d)\n", status);
          tests_failed++;
        }

        if (pool) {
          pm_free(pool, A);
          pm_free(pool, x);
          pm_free(pool, y);
        } else {
          std::free(A);
          std::free(x);
          std::free(y);
        }
      }
    }

    // Test 4: Convolution
    {
      if (verbose)
        std::printf("Test 4: Conv2D (simple 3x3)... ");

      // Simple 1x1x4x4 input, 1x1x3x3 kernel
      float input[16] = {
          1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15, 16};

      float kernel[9] = {
          1, 0, -1,
          1, 0, -1,
          1, 0, -1};

      float output[4]; // 1x1x2x2 output

      int status = mil_conv2d_f32(
          input, kernel, nullptr, output,
          1, 1, 4, 4, // batch=1, in_channels=1, 4x4 input
          1, 3, 3,    // out_channels=1, 3x3 kernel
          1, 1,       // stride
          0, 0,       // padding
          nullptr);

      if (status == MIL_OK)
      {
        if (verbose)
          std::printf("PASSED\n");
        tests_passed++;
      }
      else
      {
        if (verbose)
          std::printf("FAILED (error code: %d)\n", status);
        tests_failed++;
      }
    }

    // Summary
    if (verbose)
    {
      std::printf("\n========================================\n");
      std::printf("Tests Passed: %d\n", tests_passed);
      std::printf("Tests Failed: %d\n", tests_failed);
      std::printf("========================================\n\n");
    }

    return (tests_failed == 0) ? MIL_OK : MIL_ERR_INTERNAL;
  }

  /* ========================================================================== */
  /* Benchmark GEMM                                                              */
  /* ========================================================================== */

  int mil_benchmark_gemm(size_t min_size, size_t max_size, size_t step, int iterations)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    std::printf("\n========================================\n");
    std::printf("MIL GEMM Benchmark\n");
    std::printf("Backend: %s\n", mil_backend_name(mil_get_backend()));
    std::printf("Threads: %zu\n", mil_get_num_threads());
    std::printf("========================================\n\n");

    std::printf("%-10s %-15s %-15s %-15s\n",
                "Size", "GFLOPS", "Time(ms)", "Kernel");
    std::printf("------------------------------------------------------------------------\n");

    for (size_t size = min_size; size <= max_size; size += step)
    {
      pm_t *pool = GetUtilsPool();
      float *A = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(size * size * sizeof(float)));
      float *B = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(size * size * sizeof(float)));
      float *C = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(std::malloc(size * size * sizeof(float)));

      if (A == nullptr || B == nullptr || C == nullptr)
      {
        std::fprintf(stderr, "Allocation failed for size %zu\n", size);
        if (pool) {
          if (A) pm_free(pool, A);
          if (B) pm_free(pool, B);
          if (C) pm_free(pool, C);
        } else {
          std::free(A);
          std::free(B);
          std::free(C);
        }
        continue;
      }

      // Initialize
      for (size_t i = 0; i < size * size; ++i)
      {
        A[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        B[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        C[i] = 0.0f;
      }
#include <chrono>

      // Warmup
      mil_sgemm(MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
                size, size, size, 1.0f, A, size, B, size, 0.0f, C, size, nullptr);

      // Benchmark
      constexpr int reps = 10;
      auto t0 = std::chrono::high_resolution_clock::now();

      for (int r = 0; r < reps; ++r)
      {
        mil_sgemm(MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS,
                  size, size, size, 1.0f, A, size, B, size, 0.0f, C, size, nullptr);
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

      // Compute GFLOPS
      double flops = 2.0 * size * size * size;
      mil_perf_stats_t stats = {};
      stats.elapsed_ms = avg_ms;
      stats.gflops = (flops / 1e9) / (avg_ms / 1000.0);
      stats.kernel_used = "sgemm_benchmark";
      stats.backend_used = mil_get_backend(); // assign enum/int, not string

      mil_perf_stats_t best_stats = stats;

      std::printf("%-10zu %-15.2f %-15.3f %-15.2f %-15s\n",
                  size, best_stats.gflops, best_stats.elapsed_ms,
                  best_stats.bandwidth_gbps, best_stats.kernel_used);

      if (pool) {
        pm_free(pool, A);
        pm_free(pool, B);
        pm_free(pool, C);
      } else {
        std::free(A);
        std::free(B);
        std::free(C);
      }
      continue;
    }

    std::printf("\n");
    std::printf("========================================\n");
    std::printf("Benchmark Summary:\n");
    std::printf("  Matrix sizes tested: %zu to %zu (step %zu)\n", min_size, max_size, step);
    std::printf("  Iterations per size: %d\n", iterations);
    std::printf("  Backend: %s\n", mil_backend_name(mil_get_backend()));
    std::printf("  Threads: %zu\n", mil_get_num_threads());
    std::printf("========================================\n\n");

    return MIL_OK;
  }

} // extern "C"