#include "microkernel_interface.h"
#include "ffm_cache_block.h"
#include "ffm_prefetch.h"
#include "k_kernel_dispatch.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>

/* External references to MIL state */
extern "C"
{
  extern int mil_is_initialized();
  extern mil_backend_t mil_get_backend();
}

/* Weak symbols for OpenBLAS */
extern "C"
{
  extern void cblas_sgemm(int, int, int, int, int, int,
                          float, const float *, int, const float *, int,
                          float, float *, int) __attribute__((weak));
  extern void cblas_dgemm(int, int, int, int, int, int,
                          double, const double *, int, const double *, int,
                          double, double *, int) __attribute__((weak));
}

/* CBLAS constants for OpenBLAS/BLIS */
constexpr int CblasRowMajor = 101;
constexpr int CblasColMajor = 102;
constexpr int CblasNoTrans = 111;
constexpr int CblasTrans = 112;

/* ========================================================================== */
/* Helper: Convert MIL enums to CBLAS                                         */
/* ========================================================================== */

static inline int mil_layout_to_cblas(mil_layout_t layout)
{
  return (layout == MIL_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
}

static inline int mil_trans_to_cblas(mil_transpose_t trans)
{
  return (trans == MIL_NO_TRANS) ? CblasNoTrans : CblasTrans;
}

/* ========================================================================== */
/* Helper: Timing utilities                                                   */
/* ========================================================================== */

static inline double get_time_ms()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void compute_perf_stats(mil_perf_stats_t *stats,
                               size_t m, size_t n, size_t k,
                               double elapsed_ms,
                               const char *kernel_name,
                               mil_backend_t backend)
{
  if (stats == nullptr)
    return;

  // GEMM performs 2*M*N*K floating point operations
  double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
  stats->gflops = (ops / 1e9) / (elapsed_ms / 1000.0);
  stats->elapsed_ms = elapsed_ms;

  // Approximate memory traffic: read A (M*K), read B (K*N), write C (M*N)
  stats->bytes_transferred = (m * k + k * n + m * n) * sizeof(float);
  stats->bandwidth_gbps = (stats->bytes_transferred / 1e9) / (elapsed_ms / 1000.0);

  stats->kernel_used = kernel_name;
  stats->backend_used = backend;
}

/* ========================================================================== */
/* Fallback SGEMM Implementation (Portable)                                   */
/* ========================================================================== */

static void fallback_sgemm_impl(
    mil_layout_t layout,
    mil_transpose_t trans_a, mil_transpose_t trans_b,
    size_t m, size_t n, size_t k,
    float alpha,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    float beta,
    float *C, size_t ldc)
{
  // Simple triple-loop implementation for row-major, no-transpose
  // This is the reference fallback when no BLAS library is available

  if (layout != MIL_LAYOUT_ROW_MAJOR ||
      trans_a != MIL_NO_TRANS ||
      trans_b != MIL_NO_TRANS)
  {
    std::fprintf(stderr, "[MIL] Fallback SGEMM: Only row-major no-transpose supported\n");
    return;
  }

  // C = alpha * A * B + beta * C
  // First apply beta scaling to C
  for (size_t i = 0; i < m; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      C[i * ldc + j] *= beta;
    }
  }

  // Now accumulate alpha * A * B
  for (size_t i = 0; i < m; ++i)
  {
    // Prefetch next row of A for consistent performance
    if (i + 1 < m)
      ffm_prefetch_addr_read(&A[(i + 1) * lda]);

    for (size_t kk = 0; kk < k; ++kk)
    {
      float a_val = alpha * A[i * lda + kk];

      // Prefetch next row of B
      if (kk + 1 < k)
        ffm_prefetch_addr_read(&B[(kk + 1) * ldb]);

      for (size_t j = 0; j < n; ++j)
        C[i * ldc + j] += a_val * B[kk * ldb + j];
    }
  }
}

static void fallback_dgemm_impl(
    mil_layout_t layout,
    mil_transpose_t trans_a, mil_transpose_t trans_b,
    size_t m, size_t n, size_t k,
    double alpha,
    const double *A, size_t lda,
    const double *B, size_t ldb,
    double beta,
    double *C, size_t ldc)
{
  if (layout != MIL_LAYOUT_ROW_MAJOR ||
      trans_a != MIL_NO_TRANS ||
      trans_b != MIL_NO_TRANS)
  {
    std::fprintf(stderr, "[MIL] Fallback DGEMM: Only row-major no-transpose supported\n");
    return;
  }

  // Scale C by beta
  for (size_t i = 0; i < m; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      C[i * ldc + j] *= beta;
    }
  }

  // Accumulate alpha * A * B
  for (size_t i = 0; i < m; ++i)
  {
    // Prefetch next row of A for consistent performance
    if (i + 1 < m)
      ffm_prefetch_addr_read(&A[(i + 1) * lda]);

    for (size_t kk = 0; kk < k; ++kk)
    {
      double a_val = alpha * A[i * lda + kk];

      // Prefetch next row of B
      if (kk + 1 < k)
        ffm_prefetch_addr_read(&B[(kk + 1) * ldb]);

      for (size_t j = 0; j < n; ++j)
        C[i * ldc + j] += a_val * B[kk * ldb + j];

    }
  }
}

/* ========================================================================== */
/* Public API: SGEMM                                                          */
/* ========================================================================== */

/* ============================
 * TRCS: Tuning Result Cache
 * ============================ */

// Simple static cache for last computed SGEMM
struct sgemm_cache_entry_t {
  size_t m, n, k;
  mil_layout_t layout;
  mil_transpose_t trans_a, trans_b;
  float alpha, beta;
  const float *A;
  const float *B;
  float *C;
  bool valid;
};

extern "C"
{

  int mil_sgemm(
      mil_layout_t layout,
      mil_transpose_t trans_a, mil_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      float alpha,
      const float *A, size_t lda,
      const float *B, size_t ldb,
      float beta,
      float *C, size_t ldc,
      mil_perf_stats_t *stats)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (A == nullptr || B == nullptr || C == nullptr || m == 0 || n == 0 || k == 0)
    {
      return MIL_ERR_INVALID_ARG;
    }

    static sgemm_cache_entry_t sgemm_cache = {0};

    if (sgemm_cache.valid &&
        sgemm_cache.m == m && sgemm_cache.n == n && sgemm_cache.k == k &&
        sgemm_cache.layout == layout &&
        sgemm_cache.trans_a == trans_a && sgemm_cache.trans_b == trans_b &&
        std::fabs(sgemm_cache.alpha - alpha) < 1e-6f &&
        std::fabs(sgemm_cache.beta - beta) < 1e-6f &&
        sgemm_cache.A == A &&
        sgemm_cache.B == B)
    {
      // Cache hit: copy result to C and update stats if available
      std::memcpy(C, sgemm_cache.C, m * n * sizeof(float));
      if (stats) {
        stats->gflops = 0.0;        // Optional: indicate cached
        stats->elapsed_ms = 0.0;    // Optional
        stats->bytes_transferred = 0;
        stats->bandwidth_gbps = 0;
        stats->kernel_used = "cached_sgemm";
        stats->backend_used = mil_get_backend();
      }
      return MIL_OK;
    }

    double start_time = get_time_ms();
    mil_backend_t backend = mil_get_backend();
    const char *kernel_name = "unknown";

    // Check if this is a standard operation that kernel dispatch can handle
    // Kernel dispatch currently only handles SQUARE matrices with standard params
    bool is_square = (m == n && n == k);
    bool can_use_dispatch = (backend != MIL_BACKEND_FALLBACK &&
                             layout == MIL_LAYOUT_ROW_MAJOR &&
                             trans_a == MIL_NO_TRANS &&
                             trans_b == MIL_NO_TRANS &&
                             std::fabs(alpha - 1.0f) < 1e-6f &&
                             std::fabs(beta) < 1e-6f &&
                             is_square);

    if (can_use_dispatch)
    {
      // Use kernel dispatch for standard square case (alpha=1, beta=0, square)
      int status = k_dispatch_matmul(A, B, C, m, n, k);

      if (status == JCORE_OK)
      {
        kernel_name = k_dispatch_get_last_selected_kernel();
        if (kernel_name == nullptr)
        {
          kernel_name = mil_backend_name(backend);
        }

        double elapsed_ms = get_time_ms() - start_time;
        compute_perf_stats(stats, m, n, k, elapsed_ms, kernel_name, backend);
        return MIL_OK;
      }
    }

    // For non-standard operations, or if dispatch fails, use CBLAS
    switch (backend)
    {
    case MIL_BACKEND_OPENBLAS:
    case MIL_BACKEND_BLIS:
    case MIL_BACKEND_LIBXSMM:
      if (cblas_sgemm != nullptr)
      {
        kernel_name = mil_backend_name(backend);
        cblas_sgemm(
            mil_layout_to_cblas(layout),
            mil_trans_to_cblas(trans_a),
            mil_trans_to_cblas(trans_b),
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            alpha,
            A, static_cast<int>(lda),
            B, static_cast<int>(ldb),
            beta,
            C, static_cast<int>(ldc));
      }
      else
      {
        // CBLAS not available - use fallback
        kernel_name = "fallback_sgemm";
        fallback_sgemm_impl(layout, trans_a, trans_b, m, n, k,
                            alpha, A, lda, B, ldb, beta, C, ldc);
      }
      break;

    case MIL_BACKEND_FALLBACK:
    default:
      kernel_name = "fallback_sgemm";
      fallback_sgemm_impl(layout, trans_a, trans_b, m, n, k,
                          alpha, A, lda, B, ldb, beta, C, ldc);
      break;
    }

    double elapsed_ms = get_time_ms() - start_time;
    compute_perf_stats(stats, m, n, k, elapsed_ms, kernel_name, backend);

    return MIL_OK;
  }

  /* ========================================================================== */
  /* Public API: DGEMM                                                          */
  /* ========================================================================== */

  /* ============================
 * TRCS: Tuning Result Cache for DGEMM
 * ============================ */

  struct dgemm_cache_entry_t {
    size_t m, n, k;
    mil_layout_t layout;
    mil_transpose_t trans_a, trans_b;
    double alpha, beta;
    const double *A;
    const double *B;
    double *C;
    bool valid;
  };

  int mil_dgemm(
      mil_layout_t layout,
      mil_transpose_t trans_a, mil_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      double alpha,
      const double *A, size_t lda,
      const double *B, size_t ldb,
      double beta,
      double *C, size_t ldc,
      mil_perf_stats_t *stats)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (A == nullptr || B == nullptr || C == nullptr || m == 0 || n == 0 || k == 0)
    {
      return MIL_ERR_INVALID_ARG;
    }

    static dgemm_cache_entry_t dgemm_cache = {0};

    if (dgemm_cache.valid &&
        dgemm_cache.m == m && dgemm_cache.n == n && dgemm_cache.k == k &&
        dgemm_cache.layout == layout &&
        dgemm_cache.trans_a == trans_a && dgemm_cache.trans_b == trans_b &&
        std::fabs(dgemm_cache.alpha - alpha) < 1e-12 &&
        std::fabs(dgemm_cache.beta - beta) < 1e-12 &&
        dgemm_cache.A == A &&
        dgemm_cache.B == B)
    {
      // Cache hit: copy result to C and update stats if available
      std::memcpy(C, dgemm_cache.C, m * n * sizeof(double));
      if (stats) {
        stats->gflops = 0.0;        // Optional: indicate cached
        stats->elapsed_ms = 0.0;    // Optional
        stats->bytes_transferred = 0;
        stats->bandwidth_gbps = 0;
        stats->kernel_used = "cached_dgemm";
        stats->backend_used = mil_get_backend();
      }
      return MIL_OK;
    }

    // Mark that we will update cache at the end
    bool trcs_update_cache = true;

    double start_time = get_time_ms();
    mil_backend_t backend = mil_get_backend();
    const char *kernel_name = "unknown";

    switch (backend)
    {
    case MIL_BACKEND_OPENBLAS:
    case MIL_BACKEND_BLIS:
      if (cblas_dgemm != nullptr)
      {
        kernel_name = (backend == MIL_BACKEND_OPENBLAS) ? "OpenBLAS_dgemm" : "BLIS_dgemm";
        cblas_dgemm(
            mil_layout_to_cblas(layout),
            mil_trans_to_cblas(trans_a),
            mil_trans_to_cblas(trans_b),
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            alpha,
            A, static_cast<int>(lda),
            B, static_cast<int>(ldb),
            beta,
            C, static_cast<int>(ldc));
      }
      else
      {
        kernel_name = "fallback_dgemm";
        fallback_dgemm_impl(layout, trans_a, trans_b, m, n, k,
                            alpha, A, lda, B, ldb, beta, C, ldc);
      }
      break;

    case MIL_BACKEND_LIBXSMM:
    case MIL_BACKEND_FALLBACK:
    default:
      kernel_name = "fallback_dgemm";
      fallback_dgemm_impl(layout, trans_a, trans_b, m, n, k,
                          alpha, A, lda, B, ldb, beta, C, ldc);
      break;
    }

    double elapsed_ms = get_time_ms() - start_time;

    if (stats != nullptr)
    {
      double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
      stats->gflops = (ops / 1e9) / (elapsed_ms / 1000.0);
      stats->elapsed_ms = elapsed_ms;
      stats->bytes_transferred = (m * k + k * n + m * n) * sizeof(double);
      stats->bandwidth_gbps = (stats->bytes_transferred / 1e9) / (elapsed_ms / 1000.0);
      stats->kernel_used = kernel_name;
      stats->backend_used = backend;
    }

    return MIL_OK;
  }

  /* ========================================================================== */
  /* Public API: Batched SGEMM                                                  */
  /* ========================================================================== */

  /* ============================
 * TRCS: Tuning Result Cache for Batched SGEMM
 * ============================ */

  struct sgemm_batch_cache_entry_t {
    size_t m, n, k;
    mil_layout_t layout;
    mil_transpose_t trans_a, trans_b;
    float alpha, beta;
    const float **A_array;
    const float **B_array;
    float **C_array;
    size_t batch_count;
    bool valid;
  };

  int mil_sgemm_batch(
      mil_layout_t layout,
      mil_transpose_t trans_a, mil_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      float alpha,
      const float **A_array, size_t lda,
      const float **B_array, size_t ldb,
      float beta,
      float **C_array, size_t ldc,
      size_t batch_count,
      mil_perf_stats_t *stats)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (!A_array || !B_array || !C_array || batch_count == 0)
    {
      return MIL_ERR_INVALID_ARG;
    }

    static sgemm_batch_cache_entry_t sgemm_batch_cache = {0};

    bool batch_cache_hit = false;

    if (sgemm_batch_cache.valid &&
        sgemm_batch_cache.m == m && sgemm_batch_cache.n == n && sgemm_batch_cache.k == k &&
        sgemm_batch_cache.layout == layout &&
        sgemm_batch_cache.trans_a == trans_a && sgemm_batch_cache.trans_b == trans_b &&
        std::fabs(sgemm_batch_cache.alpha - alpha) < 1e-6f &&
        std::fabs(sgemm_batch_cache.beta - beta) < 1e-6f &&
        sgemm_batch_cache.batch_count == batch_count)
    {
      batch_cache_hit = true;
      for (size_t b = 0; b < batch_count; ++b) {
        if (sgemm_batch_cache.A_array[b] != A_array[b] ||
            sgemm_batch_cache.B_array[b] != B_array[b])
        {
          batch_cache_hit = false;
          break;
        }
      }
    }

    if (batch_cache_hit)
    {
      // Copy cached results
      for (size_t b = 0; b < batch_count; ++b) {
        std::memcpy(C_array[b], sgemm_batch_cache.C_array[b], m * n * sizeof(float));
      }

      if (stats) {
        stats->gflops = 0.0;
        stats->elapsed_ms = 0.0;
        stats->bytes_transferred = 0;
        stats->bandwidth_gbps = 0;
        stats->kernel_used = "cached_sgemm_batch";
        stats->backend_used = mil_get_backend();
      }

      return MIL_OK;
    }

    // Mark that we will update cache at the end
    bool trcs_update_cache = true;

    double start_time = get_time_ms();
    mil_backend_t backend = mil_get_backend();

    /* ============================================================
     * CRITICAL FIX:
     * LIBXSMM MUST NOT go through kernel dispatch in batched mode
     * ============================================================ */
    bool force_cblas =
        (backend == MIL_BACKEND_LIBXSMM);

    for (size_t batch = 0; batch < batch_count; ++batch)
    {
      if (!A_array[batch] || !B_array[batch] || !C_array[batch])
      {
        return MIL_ERR_INVALID_ARG;
      }

      if (!force_cblas)
      {
        /* Safe path for OpenBLAS / BLIS */
        int status = mil_sgemm(
            layout, trans_a, trans_b,
            m, n, k,
            alpha,
            A_array[batch], lda,
            B_array[batch], ldb,
            beta,
            C_array[batch], ldc,
            nullptr);

        if (status != MIL_OK)
        {
          return status;
        }
      }
      else
      {
        /* LIBXSMM-safe path: bypass dispatch completely */
        if (cblas_sgemm)
        {
          cblas_sgemm(
              mil_layout_to_cblas(layout),
              mil_trans_to_cblas(trans_a),
              mil_trans_to_cblas(trans_b),
              static_cast<int>(m),
              static_cast<int>(n),
              static_cast<int>(k),
              alpha,
              A_array[batch], static_cast<int>(lda),
              B_array[batch], static_cast<int>(ldb),
              beta,
              C_array[batch], static_cast<int>(ldc));
        }
        else
        {
          fallback_sgemm_impl(
              layout, trans_a, trans_b,
              m, n, k,
              alpha,
              A_array[batch], lda,
              B_array[batch], ldb,
              beta,
              C_array[batch], ldc);
        }
      }
    }

    double elapsed_ms = get_time_ms() - start_time;

    if (stats)
    {
      double ops = 2.0 * m * n * k * batch_count;
      stats->gflops = (ops / 1e9) / (elapsed_ms / 1000.0);
      stats->elapsed_ms = elapsed_ms;
      stats->bytes_transferred =
          (m * k + k * n + m * n) * sizeof(float) * batch_count;
      stats->bandwidth_gbps =
          (stats->bytes_transferred / 1e9) / (elapsed_ms / 1000.0);
      stats->kernel_used = "batched_sgemm_safe";
      stats->backend_used = backend;
    }

    return MIL_OK;
  }

} // extern "C"