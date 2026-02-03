#ifndef JCORE_MICROKERNEL_INTERFACE_H
#define JCORE_MICROKERNEL_INTERFACE_H

/**
 * @file microkernel_interface.h
 * @brief Microkernel Interface Layer - Unified calling convention for GEMM/Conv kernels
 *
 * Component: Microkernel Interface Layer (Derived)
 * Purpose: Unified calling convention wrapping BLAS (OpenBLAS/BLIS) and LIBXSMM
 * Dependencies:
 *   - Kernel Dispatch Table/Runtime Selector (derived)
 *   - Vector Math Engine (derived)
 *   - CPU Feature Detection Module (base)
 *   - ISA-Aware Dispatch Mechanism (base)
 *   - Cache Blocking/Tiling Utility (base)
 *   - Memory Prefetch Interface (base)
 *   - Thread Scheduler (base)
 *
 * Design:
 *   - Runtime backend selection (OpenBLAS, BLIS, LIBXSMM, fallback)
 *   - Unified API for GEMM, GEMV, Conv2D operations
 *   - Automatic tile/block size computation for cache efficiency
 *   - Thread-safe, FFM-compatible C API
 *   - Performance monitoring and kernel selection
 *
 * Thread-safety: All functions are thread-safe after initialization
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ========================================================================== */
  /* Error Codes                                                                 */
  /* ========================================================================== */

#define MIL_OK 0
#define MIL_ERR_NOT_INITIALIZED -1
#define MIL_ERR_INVALID_ARG -2
#define MIL_ERR_NO_BACKEND -3
#define MIL_ERR_INTERNAL -4
#define MIL_ERR_ALLOCATION -5
#define MIL_ERR_UNSUPPORTED -6

  /* ========================================================================== */
  /* Backend Selection                                                           */
  /* ========================================================================== */

  typedef enum
  {
    MIL_BACKEND_AUTO = 0,     /**< Auto-select best available backend */
    MIL_BACKEND_OPENBLAS = 1, /**< OpenBLAS library */
    MIL_BACKEND_BLIS = 2,     /**< BLIS library */
    MIL_BACKEND_LIBXSMM = 3,  /**< LIBXSMM (JIT kernels) */
    MIL_BACKEND_FALLBACK = 4  /**< Portable fallback implementation */
  } mil_backend_t;

  /* ========================================================================== */
  /* Matrix Layout                                                               */
  /* ========================================================================== */

  typedef enum
  {
    MIL_LAYOUT_ROW_MAJOR = 0, /**< Row-major (C-style) */
    MIL_LAYOUT_COL_MAJOR = 1  /**< Column-major (Fortran-style) */
  } mil_layout_t;

  /* ========================================================================== */
  /* Transpose Operations                                                        */
  /* ========================================================================== */

  typedef enum
  {
    MIL_NO_TRANS = 0,  /**< No transpose */
    MIL_TRANS = 1,     /**< Transpose */
    MIL_CONJ_TRANS = 2 /**< Conjugate transpose (for complex) */
  } mil_transpose_t;

  /* ========================================================================== */
  /* Data Types                                                                  */
  /* ========================================================================== */

  typedef enum
  {
    MIL_DTYPE_FP32 = 0, /**< Single precision float */
    MIL_DTYPE_FP64 = 1, /**< Double precision double */
    MIL_DTYPE_INT8 = 2, /**< 8-bit integer (quantized) */
    MIL_DTYPE_INT16 = 3 /**< 16-bit integer */
  } mil_dtype_t;

  /* ========================================================================== */
  /* Configuration Structure                                                     */
  /* ========================================================================== */

  typedef struct
  {
    mil_backend_t preferred_backend; /**< Preferred backend (AUTO = auto-select) */
    size_t num_threads;              /**< Number of threads (0 = auto) */
    int enable_prefetch;             /**< Enable memory prefetching (1/0) */
    int enable_auto_tuning;          /**< Enable adaptive kernel selection (1/0) */
    int verbose;                     /**< Verbose logging (1/0) */
  } mil_config_t;

  /* ========================================================================== */
  /* Performance Statistics                                                      */
  /* ========================================================================== */

  typedef struct
  {
    double gflops;              /**< Achieved GFLOPS */
    double elapsed_ms;          /**< Elapsed time in milliseconds */
    size_t bytes_transferred;   /**< Bytes read/written */
    double bandwidth_gbps;      /**< Memory bandwidth in GB/s */
    const char *kernel_used;    /**< Name of kernel used */
    mil_backend_t backend_used; /**< Backend that executed operation */
  } mil_perf_stats_t;

  /* ========================================================================== */
  /* Initialization & Configuration                                              */
  /* ========================================================================== */

  /**
   * @brief Initialize Microkernel Interface Layer
   *
   * Detects available BLAS libraries, initializes backends, and sets up
   * the dispatch table. Must be called before any MIL operations.
   *
   * @param config Configuration structure (NULL = use defaults)
   * @return MIL_OK on success, error code otherwise
   */
  int mil_init(const mil_config_t *config);

  /**
   * @brief Shutdown Microkernel Interface Layer
   *
   * Cleanup resources and finalize backends. Safe to call multiple times.
   */
  void mil_shutdown(void);

  /**
   * @brief Check if MIL is initialized
   *
   * @return 1 if initialized, 0 otherwise
   */
  int mil_is_initialized(void);

  /**
   * @brief Get current active backend
   *
   * @return Current backend in use
   */
  mil_backend_t mil_get_backend(void);

  /**
   * @brief Get backend name string
   *
   * @param backend Backend enum
   * @return Human-readable name (e.g., "OpenBLAS")
   */
  const char *mil_backend_name(mil_backend_t backend);

  /**
   * @brief Set number of threads for BLAS operations
   *
   * @param num_threads Number of threads (0 = auto)
   * @return MIL_OK on success
   */
  int mil_set_num_threads(size_t num_threads);

  /**
   * @brief Get current thread count
   *
   * @return Number of threads in use
   */
  size_t mil_get_num_threads(void);

  /* ========================================================================== */
  /* GEMM Operations (General Matrix Multiply)                                  */
  /* ========================================================================== */

  /**
   * @brief Single precision GEMM: C = alpha * op(A) * op(B) + beta * C
   *
   * @param layout Matrix storage layout
   * @param trans_a Transpose operation for A
   * @param trans_b Transpose operation for B
   * @param m Number of rows in op(A) and C
   * @param n Number of columns in op(B) and C
   * @param k Number of columns in op(A) and rows in op(B)
   * @param alpha Scalar alpha
   * @param A Matrix A
   * @param lda Leading dimension of A
   * @param B Matrix B
   * @param ldb Leading dimension of B
   * @param beta Scalar beta
   * @param C Matrix C (output)
   * @param ldc Leading dimension of C
   * @param stats Optional performance statistics (NULL = don't collect)
   * @return MIL_OK on success
   */
  int mil_sgemm(
      mil_layout_t layout,
      mil_transpose_t trans_a, mil_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      float alpha,
      const float *A, size_t lda,
      const float *B, size_t ldb,
      float beta,
      float *C, size_t ldc,
      mil_perf_stats_t *stats);

  /**
   * @brief Double precision GEMM: C = alpha * op(A) * op(B) + beta * C
   */
  int mil_dgemm(
      mil_layout_t layout,
      mil_transpose_t trans_a, mil_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      double alpha,
      const double *A, size_t lda,
      const double *B, size_t ldb,
      double beta,
      double *C, size_t ldc,
      mil_perf_stats_t *stats);

  /**
   * @brief Batched SGEMM: Perform multiple GEMM operations
   *
   * @param layout Matrix storage layout
   * @param trans_a Transpose operation for A
   * @param trans_b Transpose operation for B
   * @param m Number of rows in op(A) and C
   * @param n Number of columns in op(B) and C
   * @param k Number of columns in op(A) and rows in op(B)
   * @param alpha Scalar alpha
   * @param A_array Array of matrix A pointers
   * @param lda Leading dimension of A
   * @param B_array Array of matrix B pointers
   * @param ldb Leading dimension of B
   * @param beta Scalar beta
   * @param C_array Array of matrix C pointers (output)
   * @param ldc Leading dimension of C
   * @param batch_count Number of GEMM operations in batch
   * @param stats Optional performance statistics (NULL = don't collect)
   * @return MIL_OK on success
   */
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
      mil_perf_stats_t *stats);

  /* ========================================================================== */
  /* GEMV Operations (General Matrix-Vector Multiply)                           */
  /* ========================================================================== */

  /**
   * @brief Single precision GEMV: y = alpha * op(A) * x + beta * y
   *
   * @param layout Matrix storage layout
   * @param trans Transpose operation for A
   * @param m Number of rows in A
   * @param n Number of columns in A
   * @param alpha Scalar alpha
   * @param A Matrix A
   * @param lda Leading dimension of A
   * @param x Vector x
   * @param incx Stride of x
   * @param beta Scalar beta
   * @param y Vector y (output)
   * @param incy Stride of y
   * @param stats Optional performance statistics
   * @return MIL_OK on success
   */
  int mil_sgemv(
      mil_layout_t layout,
      mil_transpose_t trans,
      size_t m, size_t n,
      float alpha,
      const float *A, size_t lda,
      const float *x, int incx,
      float beta,
      float *y, int incy,
      mil_perf_stats_t *stats);

  /**
   * @brief Double precision GEMV: y = alpha * op(A) * x + beta * y
   */
  int mil_dgemv(
      mil_layout_t layout,
      mil_transpose_t trans,
      size_t m, size_t n,
      double alpha,
      const double *A, size_t lda,
      const double *x, int incx,
      double beta,
      double *y, int incy,
      mil_perf_stats_t *stats);

  /* ========================================================================== */
  /* Convolution Operations                                                      */
  /* ========================================================================== */

  /**
   * @brief 2D Convolution (direct algorithm)
   *
   * Computes: output = conv2d(input, kernel) + bias
   *
   * @param input Input tensor [batch, in_channels, in_h, in_w]
   * @param kernel Convolution kernel [out_channels, in_channels, kh, kw]
   * @param bias Bias vector [out_channels] (can be NULL)
   * @param output Output tensor [batch, out_channels, out_h, out_w]
   * @param batch Batch size
   * @param in_channels Input channels
   * @param in_h Input height
   * @param in_w Input width
   * @param out_channels Output channels
   * @param kh Kernel height
   * @param kw Kernel width
   * @param stride_h Vertical stride
   * @param stride_w Horizontal stride
   * @param pad_h Vertical padding
   * @param pad_w Horizontal padding
   * @param stats Optional performance statistics
   * @return MIL_OK on success
   */
  int mil_conv2d_f32(
      const float *input,
      const float *kernel,
      const float *bias,
      float *output,
      size_t batch,
      size_t in_channels,
      size_t in_h, size_t in_w,
      size_t out_channels,
      size_t kh, size_t kw,
      size_t stride_h, size_t stride_w,
      size_t pad_h, size_t pad_w,
      mil_perf_stats_t *stats);

  /**
   * @brief 2D Convolution using im2col + GEMM
   *
   * More efficient for large kernels. Transforms convolution into matrix multiply.
   *
   * @param input Input tensor
   * @param kernel Convolution kernel
   * @param bias Bias vector (can be NULL)
   * @param output Output tensor
   * @param batch Batch size
   * @param in_channels Input channels
   * @param in_h Input height
   * @param in_w Input width
   * @param out_channels Output channels
   * @param kh Kernel height
   * @param kw Kernel width
   * @param stride_h Vertical stride
   * @param stride_w Horizontal stride
   * @param pad_h Vertical padding
   * @param pad_w Horizontal padding
   * @param stats Optional performance statistics
   * @return MIL_OK on success
   */
  int mil_conv2d_im2col_f32(
      const float *input,
      const float *kernel,
      const float *bias,
      float *output,
      size_t batch,
      size_t in_channels,
      size_t in_h, size_t in_w,
      size_t out_channels,
      size_t kh, size_t kw,
      size_t stride_h, size_t stride_w,
      size_t pad_h, size_t pad_w,
      mil_perf_stats_t *stats);

/* ========================================================================== */
/* Vector Operations (Single Precision - float)                               */
/* ========================================================================== */

  /**
   * @brief Compute exp(x) for vector of floats by dispatching VMath Engine Internally
   * @param x Input array
   * @param y Output array (can alias x for in-place)
   * @param n Number of elements
   * @return MIL_OK or error code
   */
  int mil_expf(const float *x, float *y, size_t n);

  /**
   * @brief Compute natural log ln(x) for vector of floats by dispatching VMath Engine Internally
   * @param x Input array (must be > 0)
   * @param y Output array
   * @param n Number of elements
   * @return MIL_OK or error code
   */
  int mil_logf(const float *x, float *y, size_t n);

  /**
   * @brief Compute log10(x) for vector of floats by dispatching VMath Engine Internally
   */
  int mil_log10f(const float *x, float *y, size_t n);

  /**
   * @brief Compute sine sin(x) for vector of floats by dispatching VMath Engine Internally
   * @param x Input array (radians)
   * @param y Output array
   * @param n Number of elements
   */
  int mil_sinf(const float *x, float *y, size_t n);

  /**
   * @brief Compute cosine cos(x) by dispatching VMath Engine Internally
   */
  int mil_cosf(const float *x, float *y, size_t n);

  /**
   * @brief Compute tangent tan(x) by dispatching VMath Engine Internally
   */
  int mil_tanf(const float *x, float *y, size_t n);

  /**
   * @brief Compute arc sine asin(x) (domain [-1, 1]) by dispatching VMath Engine Internally
   */
  int mil_asinf(const float *x, float *y, size_t n);

  /**
   * @brief Compute arc cosine acos(x) by dispatching VMath Engine Internally
   */
  int mil_acosf(const float *x, float *y, size_t n);

  /**
   * @brief Compute arc tangent atan(x) by dispatching VMath Engine Internally
   */
  int mil_atanf(const float *x, float *y, size_t n);

  /**
   * @brief Compute two-argument arc tangent atan2(y, x) by dispatching VMath Engine Internally
   * @param y Numerator array
   * @param x Denominator array
   * @param result Output array
   * @param n Number of elements
   */
  int mil_atan2f(const float *y, const float *x, float *result, size_t n);

  /**
   * @brief Compute hyperbolic sine sinh(x) by dispatching VMath Engine Internally
   */
  int mil_sinhf(const float *x, float *y, size_t n);

  /**
   * @brief Compute hyperbolic cosine cosh(x) by dispatching VMath Engine Internally
   */
  int mil_coshf(const float *x, float *y, size_t n);

  /**
   * @brief Compute hyperbolic tangent tanh(x) by dispatching VMath Engine Internally
   */
  int mil_tanhf(const float *x, float *y, size_t n);

  /**
   * @brief Compute square root sqrt(x) by dispatching VMath Engine Internally
   */
  int mil_sqrtf(const float *x, float *y, size_t n);

  /**
   * @brief Compute cube root cbrt(x) by dispatching VMath Engine Internally
   */
  int mil_cbrtf(const float *x, float *y, size_t n);

  /**
   * @brief Compute power x^p (scalar exponent) by dispatching VMath Engine Internally
   * @param x Base array
   * @param p Exponent (scalar)
   * @param y Output array
   * @param n Number of elements
   */
  int mil_powf_scalar(const float *x, float p, float *y, size_t n);

  /**
   * @brief Compute power x^y (vector exponent) by dispatching VMath Engine Internally
   * @param x Base array
   * @param y Exponent array
   * @param result Output array
   * @param n Number of elements
   */
  int mil_powf(const float *x, const float *y, float *result, size_t n);

  /**
   * @brief Compute error function erf(x) by dispatching VMath Engine Internally
   */
  int mil_erff(const float *x, float *y, size_t n);

  /**
   * @brief Compute complementary error function erfc(x) by dispatching VMath Engine Internally
   */
  int mil_erfcf(const float *x, float *y, size_t n);

  /**
   * @brief Compute gamma function Γ(x) by dispatching VMath Engine Internally
   */
  int mil_gammaf(const float *x, float *y, size_t n);

  /**
   * @brief Compute natural log of gamma function ln(Γ(x)) by dispatching VMath Engine Internally
   */
  int mil_lgammaf(const float *x, float *y, size_t n);

/* ========================================================================== */
/* Vector Operations (Double Precision - double)                              */
/* ========================================================================== */

  int mil_exp(const double *x, double *y, size_t n);
  int mil_log(const double *x, double *y, size_t n);
  int mil_log10(const double *x, double *y, size_t n);
  int mil_sin(const double *x, double *y, size_t n);
  int mil_cos(const double *x, double *y, size_t n);
  int mil_tan(const double *x, double *y, size_t n);
  int mil_asin(const double *x, double *y, size_t n);
  int mil_acos(const double *x, double *y, size_t n);
  int mil_atan(const double *x, double *y, size_t n);
  int mil_atan2(const double *y, const double *x, double *result, size_t n);
  int mil_sinh(const double *x, double *y, size_t n);
  int mil_cosh(const double *x, double *y, size_t n);
  int mil_tanh(const double *x, double *y, size_t n);
  int mil_sqrt(const double *x, double *y, size_t n);
  int mil_cbrt(const double *x, double *y, size_t n);
  int mil_pow_scalar(const double *x, double p, double *y, size_t n);
  int mil_pow(const double *x, const double *y, double *result, size_t n);
  int mil_erf(const double *x, double *y, size_t n);
  int mil_erfc(const double *x, double *y, size_t n);
  int mil_gamma(const double *x, double *y, size_t n);
  int mil_lgamma(const double *x, double *y, size_t n);

  /* ========================================================================== */
  /* Fused Operations (Common ML/Scientific patterns)                           */
  /* ========================================================================== */

  /**
   * @brief Compute sigmoid: 1 / (1 + exp(-x)) by dispatching VMath Engine Internally
   */
  int mil_sigmoidf(const float *x, float *y, size_t n);
  int mil_sigmoid(const double *x, double *y, size_t n);

  /**
   * @brief Compute ReLU: max(0, x) by dispatching VMath Engine Internally
   */
  int mil_reluf(const float *x, float *y, size_t n);
  int mil_relu(const double *x, double *y, size_t n);


  /**
   * @brief Compute Leaky ReLU: x if x > 0, else alpha*x (alpha=0.01)
   */
  int mil_leaky_reluf(const float *x, float *y, size_t n);
  int mil_leaky_relu(const double *x, double *y, size_t n);

  /**
    * @brief Compute ReLU6: min(max(x, 0), 6)
    */
  int mil_relu6f(const float *x, float *y, size_t n);
  int mil_relu6(const double *x, double *y, size_t n);

  /**
   * @brief Compute softplus: log(1 + exp(x)) by dispatching VMath Engine Internally
   */
  int mil_softplusf(const float *x, float *y, size_t n);
  int mil_softplus(const double *x, double *y, size_t n);

  /**
   * @brief Compute GELU: x * Φ(x) where Φ is standard normal CDF by dispatching VMath Engine Internally
   * Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
   */
  int mil_geluf(const float *x, float *y, size_t n);
  int mil_gelu(const double *x, double *y, size_t n);

/* ========================================================================== */
/* Utility Functions                                                           */
/* ========================================================================== */

  /**
   * @brief Compute optimal tile sizes for current cache hierarchy
   *
   * @param m Matrix dimension M
   * @param n Matrix dimension N
   * @param k Matrix dimension K
   * @param elem_size Size of element in bytes (e.g., 4 for float)
   * @param tile_m Output: optimal tile size for M dimension
   * @param tile_n Output: optimal tile size for N dimension
   * @param tile_k Output: optimal tile size for K dimension
   * @return MIL_OK on success
   */
  int mil_compute_optimal_tiles(
      size_t m, size_t n, size_t k,
      size_t elem_size,
      size_t *tile_m, size_t *tile_n, size_t *tile_k);

  /**
   * @brief Get system information string
   *
   * @return Static string with CPU features, cache sizes, backend info
   */
  const char *mil_get_system_info(void);

  /**
   * @brief Convert error code to human-readable string
   *
   * @param error Error code
   * @return Error message string
   */
  const char *mil_strerror(int error);

  /**
   * @brief Run comprehensive self-test
   *
   * Tests all major operations across available backends.
   *
   * @param verbose Print detailed results (1/0)
   * @return MIL_OK if all tests pass
   */
  int mil_self_test(int verbose);

  /**
   * @brief Benchmark GEMM performance across problem sizes
   *
   * @param min_size Minimum matrix dimension
   * @param max_size Maximum matrix dimension
   * @param step Step size for dimensions
   * @param iterations Number of iterations per size
   * @return MIL_OK on success
   */
  int mil_benchmark_gemm(size_t min_size, size_t max_size, size_t step, int iterations);

#ifdef __cplusplus
}
#endif

#endif /* JCORE_MICROKERNEL_INTERFACE_H */