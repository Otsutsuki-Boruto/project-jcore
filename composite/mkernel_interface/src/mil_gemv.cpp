#include "microkernel_interface.h"
#include "ffm_prefetch.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

/* External references */
extern "C"
{
  extern int mil_is_initialized();
  extern mil_backend_t mil_get_backend();
}

/* Weak symbols for CBLAS */
extern "C"
{
  extern void cblas_sgemv(int, int, int, int, float,
                          const float *, int, const float *, int,
                          float, float *, int) __attribute__((weak));
  extern void cblas_dgemv(int, int, int, int, double,
                          const double *, int, const double *, int,
                          double, double *, int) __attribute__((weak));
}

constexpr int CblasRowMajor = 101;
constexpr int CblasColMajor = 102;
constexpr int CblasNoTrans = 111;
constexpr int CblasTrans = 112;

/* ========================================================================== */
/* Helper: Timing                                                              */
/* ========================================================================== */

static inline double get_time_ms()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ========================================================================== */
/* Helper: Convert enums                                                      */
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
/* Fallback SGEMV Implementation                                              */
/* ========================================================================== */

static void fallback_sgemv_impl(
    mil_layout_t layout,
    mil_transpose_t trans,
    size_t m, size_t n,
    float alpha,
    const float *A, size_t lda,
    const float *x, int incx,
    float beta,
    float *y, int incy)
{
  // Only support row-major, no-transpose for simplicity
  if (layout != MIL_LAYOUT_ROW_MAJOR || trans != MIL_NO_TRANS)
  {
    std::fprintf(stderr, "[MIL] Fallback SGEMV: Only row-major no-transpose supported\n");
    return;
  }

  // y = alpha * A * x + beta * y
  // A is m x n, x is n x 1, y is m x 1

  // First scale y by beta
  for (size_t i = 0; i < m; ++i)
  {
    y[i * incy] *= beta;
  }

  // Accumulate alpha * A * x
  for (size_t i = 0; i < m; ++i)
  {
    // Prefetch next row for consistent performance
    if (i + 1 < m)
    {
      ffm_prefetch_addr_read(&A[(i + 1) * lda]);
    }

    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j)
    {
      sum += A[i * lda + j] * x[j * incx];
    }
    y[i * incy] += alpha * sum;
  }
}

static void fallback_dgemv_impl(
    mil_layout_t layout,
    mil_transpose_t trans,
    size_t m, size_t n,
    double alpha,
    const double *A, size_t lda,
    const double *x, int incx,
    double beta,
    double *y, int incy)
{
  if (layout != MIL_LAYOUT_ROW_MAJOR || trans != MIL_NO_TRANS)
  {
    std::fprintf(stderr, "[MIL] Fallback DGEMV: Only row-major no-transpose supported\n");
    return;
  }

  // Scale y by beta
  for (size_t i = 0; i < m; ++i)
  {
    y[i * incy] *= beta;
  }

  // Accumulate alpha * A * x
  for (size_t i = 0; i < m; ++i)
  {
    // Prefetch next row for consistent performance
    if (i + 1 < m)
    {
      ffm_prefetch_addr_read(&A[(i + 1) * lda]);
    }

    double sum = 0.0;
    for (size_t j = 0; j < n; ++j)
    {
      sum += A[i * lda + j] * x[j * incx];
    }
    y[i * incy] += alpha * sum;
  }
}

/* ========================================================================== */
/* Public API: SGEMV                                                          */
/* ========================================================================== */

extern "C"
{

  int mil_sgemv(
      mil_layout_t layout,
      mil_transpose_t trans,
      size_t m, size_t n,
      float alpha,
      const float *A, size_t lda,
      const float *x, int incx,
      float beta,
      float *y, int incy,
      mil_perf_stats_t *stats)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (A == nullptr || x == nullptr || y == nullptr || m == 0 || n == 0)
    {
      return MIL_ERR_INVALID_ARG;
    }

    double start_time = get_time_ms();
    mil_backend_t backend = mil_get_backend();
    const char *kernel_name = "unknown";

    switch (backend)
    {
    case MIL_BACKEND_OPENBLAS:
    case MIL_BACKEND_BLIS:
      if (cblas_sgemv != nullptr)
      {
        kernel_name = (backend == MIL_BACKEND_OPENBLAS) ? "OpenBLAS_sgemv" : "BLIS_sgemv";
        cblas_sgemv(
            mil_layout_to_cblas(layout),
            mil_trans_to_cblas(trans),
            static_cast<int>(m), static_cast<int>(n),
            alpha,
            A, static_cast<int>(lda),
            x, incx,
            beta,
            y, incy);
      }
      else
      {
        kernel_name = "fallback_sgemv";
        fallback_sgemv_impl(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
      }
      break;

    case MIL_BACKEND_LIBXSMM:
    case MIL_BACKEND_FALLBACK:
    default:
      kernel_name = "fallback_sgemv";
      fallback_sgemv_impl(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
      break;
    }

    double elapsed_ms = get_time_ms() - start_time;

    if (stats != nullptr)
    {
      // GEMV performs 2*M*N operations
      double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n);
      stats->gflops = (ops / 1e9) / (elapsed_ms / 1000.0);
      stats->elapsed_ms = elapsed_ms;
      stats->bytes_transferred = (m * n + n + m) * sizeof(float);
      stats->bandwidth_gbps = (stats->bytes_transferred / 1e9) / (elapsed_ms / 1000.0);
      stats->kernel_used = kernel_name;
      stats->backend_used = backend;
    }

    return MIL_OK;
  }

  /* ========================================================================== */
  /* Public API: DGEMV                                                          */
  /* ========================================================================== */

  int mil_dgemv(
      mil_layout_t layout,
      mil_transpose_t trans,
      size_t m, size_t n,
      double alpha,
      const double *A, size_t lda,
      const double *x, int incx,
      double beta,
      double *y, int incy,
      mil_perf_stats_t *stats)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (A == nullptr || x == nullptr || y == nullptr || m == 0 || n == 0)
    {
      return MIL_ERR_INVALID_ARG;
    }

    double start_time = get_time_ms();
    mil_backend_t backend = mil_get_backend();
    const char *kernel_name = "unknown";

    switch (backend)
    {
    case MIL_BACKEND_OPENBLAS:
    case MIL_BACKEND_BLIS:
      if (cblas_dgemv != nullptr)
      {
        kernel_name = (backend == MIL_BACKEND_OPENBLAS) ? "OpenBLAS_dgemv" : "BLIS_dgemv";
        cblas_dgemv(
            mil_layout_to_cblas(layout),
            mil_trans_to_cblas(trans),
            static_cast<int>(m), static_cast<int>(n),
            alpha,
            A, static_cast<int>(lda),
            x, incx,
            beta,
            y, incy);
      }
      else
      {
        kernel_name = "fallback_dgemv";
        fallback_dgemv_impl(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
      }
      break;

    case MIL_BACKEND_LIBXSMM:
    case MIL_BACKEND_FALLBACK:
    default:
      kernel_name = "fallback_dgemv";
      fallback_dgemv_impl(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
      break;
    }

    double elapsed_ms = get_time_ms() - start_time;

    if (stats != nullptr)
    {
      double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n);
      stats->gflops = (ops / 1e9) / (elapsed_ms / 1000.0);
      stats->elapsed_ms = elapsed_ms;
      stats->bytes_transferred = (m * n + n + m) * sizeof(double);
      stats->bandwidth_gbps = (stats->bytes_transferred / 1e9) / (elapsed_ms / 1000.0);
      stats->kernel_used = kernel_name;
      stats->backend_used = backend;
    }

    return MIL_OK;
  }

} // extern "C"