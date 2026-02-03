// advanced/kFusion_engine/include/kernel_fusion_engine.h
#ifndef JCORE_KERNEL_FUSION_ENGINE_H_
#define JCORE_KERNEL_FUSION_ENGINE_H_

/**
 * @file kernel_fusion_engine.h
 * @brief Kernel Fusion Engine - Fuses ops (GEMM + Bias + ReLU) to eliminate memory round-trips
 *
 * Component: Kernel Fusion Engine (Advanced)
 * Purpose: Eliminate intermediate memory traffic by fusing computational kernels
 *
 * Dependencies:
 *   Derived:
 *     - Microkernel Interface Layer: Core GEMM operations
 *     - Adaptive Kernel Auto-Tuner: Optimal kernel selection
 *     - Memory Pool Manager: Workspace allocation
 *   Base:
 *     - Cache Blocking/Tiling Utility: Optimal tile sizes
 *     - Memory Allocator Wrapper: Aligned allocations
 *     - ISA-Aware Dispatch: CPU feature detection
 *     - Thread Scheduler: Parallel execution
 *
 * Supported Fusion Patterns:
 *   - GEMM + Bias
 *   - GEMM + Bias + ReLU
 *   - GEMM + Bias + Tanh
 *   - GEMM + Bias + Sigmoid
 *   - GEMM + Bias + GELU
 *   - GEMM + Elementwise Add/Mul
 *
 * Performance Benefits:
 *   - Eliminates intermediate buffer allocation
 *   - Reduces memory bandwidth pressure
 *   - Improves cache utilization
 *   - Enables vectorized activation functions
 *
 * Thread-safety: Thread-safe after initialization
 * FFM API: Fully compatible with Project JCore FFM API
 */

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ========================================================================== */
  /* Error Codes                                                                 */
  /* ========================================================================== */

#define KFE_OK 0
#define KFE_ERR_NOT_INITIALIZED -1
#define KFE_ERR_INVALID_ARG -2
#define KFE_ERR_NO_MEMORY -3
#define KFE_ERR_INTERNAL -4
#define KFE_ERR_UNSUPPORTED -5
#define KFE_ERR_ALLOCATION -6

  /* ========================================================================== */
  /* Activation Function Types                                                   */
  /* ========================================================================== */

  typedef enum
  {
    KFE_ACTIVATION_NONE = 0,      /**< No activation (linear) */
    KFE_ACTIVATION_RELU = 1,      /**< ReLU: max(0, x) */
    KFE_ACTIVATION_RELU6 = 2,     /**< ReLU6: min(max(0, x), 6) */
    KFE_ACTIVATION_TANH = 3,      /**< Tanh: (e^x - e^-x)/(e^x + e^-x) */
    KFE_ACTIVATION_SIGMOID = 4,   /**< Sigmoid: 1/(1 + e^-x) */
    KFE_ACTIVATION_GELU = 5,      /**< GELU: 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x^3))) */
    KFE_ACTIVATION_SWISH = 6,     /**< Swish: x * sigmoid(x) */
    KFE_ACTIVATION_LEAKY_RELU = 7 /**< Leaky ReLU: max(0.01*x, x) */
  } kfe_activation_t;

  /* ========================================================================== */
  /* Fusion Operation Types                                                      */
  /* ========================================================================== */

  typedef enum
  {
    KFE_FUSION_GEMM_BIAS = 1,            /**< C = alpha*A*B + bias (broadcast) */
    KFE_FUSION_GEMM_BIAS_ACTIVATION = 2, /**< C = activation(alpha*A*B + bias) */
    KFE_FUSION_GEMM_ELEMENTWISE_ADD = 3, /**< C = alpha*A*B + beta*D (element-wise) */
    KFE_FUSION_GEMM_ELEMENTWISE_MUL = 4, /**< C = (alpha*A*B) * D (element-wise) */
    KFE_FUSION_GEMM_RESIDUAL = 5         /**< C = activation(alpha*A*B + bias + residual) */
  } kfe_fusion_type_t;

  /* ========================================================================== */
  /* Matrix Layout                                                               */
  /* ========================================================================== */

  typedef enum
  {
    KFE_LAYOUT_ROW_MAJOR = 0, /**< Row-major (C-style) */
    KFE_LAYOUT_COL_MAJOR = 1  /**< Column-major (Fortran-style) */
  } kfe_layout_t;

  /* ========================================================================== */
  /* Transpose Operations                                                        */
  /* ========================================================================== */

  typedef enum
  {
    KFE_NO_TRANS = 0, /**< No transpose */
    KFE_TRANS = 1     /**< Transpose */
  } kfe_transpose_t;

  /* ========================================================================== */
  /* Configuration Structure                                                     */
  /* ========================================================================== */

  typedef struct
  {
    size_t num_threads;           /**< Number of threads (0 = auto) */
    int enable_vectorization;     /**< Enable SIMD vectorization (1/0) */
    int enable_cache_blocking;    /**< Enable cache-aware tiling (1/0) */
    int enable_prefetch;          /**< Enable memory prefetching (1/0) */
    int enable_kernel_autotuning; /**< Use adaptive kernel selection (1/0) */
    size_t workspace_size_mb;     /**< Workspace pool size in MB (0 = auto) */
    int verbose;                  /**< Verbose logging (1/0) */
  } kfe_config_t;

  /* ========================================================================== */
  /* Performance Statistics                                                      */
  /* ========================================================================== */

  typedef struct
  {
    double gflops;              /**< Achieved GFLOPS */
    double elapsed_ms;          /**< Elapsed time in milliseconds */
    size_t fused_ops_count;     /**< Number of fused operations */
    size_t memory_saved_bytes;  /**< Memory traffic eliminated */
    double bandwidth_gbps;      /**< Memory bandwidth in GB/s */
    const char *fusion_pattern; /**< Fusion pattern applied */
    const char *kernel_backend; /**< Backend used (OpenBLAS/BLIS/LIBXSMM) */
  } kfe_perf_stats_t;

  /* ========================================================================== */
  /* Initialization & Configuration                                              */
  /* ========================================================================== */

  /**
   * @brief Initialize Kernel Fusion Engine
   *
   * Initializes all dependencies (MIL, Pool Manager, Tuner, etc.)
   * and prepares fusion runtime.
   *
   * @param config Configuration structure (NULL = use defaults)
   * @return KFE_OK on success, error code otherwise
   */
  int kfe_init(const kfe_config_t *config);

  /**
   * @brief Shutdown Kernel Fusion Engine
   *
   * Cleanup resources and finalize all subsystems.
   * Safe to call multiple times.
   */
  void kfe_shutdown(void);

  /**
   * @brief Check if KFE is initialized
   *
   * @return 1 if initialized, 0 otherwise
   */
  int kfe_is_initialized(void);

  /**
   * @brief Set number of threads for fused operations
   *
   * @param num_threads Number of threads (0 = auto)
   * @return KFE_OK on success
   */
  int kfe_set_num_threads(size_t num_threads);

  /**
   * @brief Get current thread count
   *
   * @return Number of threads in use
   */
  size_t kfe_get_num_threads(void);

  /* ========================================================================== */
  /* Fused GEMM Operations                                                       */
  /* ========================================================================== */

  /**
   * @brief Fused GEMM + Bias: C = alpha * op(A) * op(B) + bias (broadcasted)
   *
   * Bias vector is broadcasted across all rows. Eliminates intermediate buffer.
   *
   * @param layout Matrix storage layout
   * @param trans_a Transpose operation for A
   * @param trans_b Transpose operation for B
   * @param m Number of rows in op(A) and C
   * @param n Number of columns in op(B) and C
   * @param k Number of columns in op(A) and rows in op(B)
   * @param alpha Scalar alpha
   * @param A Matrix A [m x k or k x m if transposed]
   * @param lda Leading dimension of A
   * @param B Matrix B [k x n or n x k if transposed]
   * @param ldb Leading dimension of B
   * @param bias Bias vector [n] (broadcasted across rows)
   * @param C Matrix C (output) [m x n]
   * @param ldc Leading dimension of C
   * @param stats Optional performance statistics (NULL = don't collect)
   * @return KFE_OK on success
   */
  int kfe_sgemm_bias(
      kfe_layout_t layout,
      kfe_transpose_t trans_a, kfe_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      float alpha,
      const float *A, size_t lda,
      const float *B, size_t ldb,
      const float *bias,
      float *C, size_t ldc,
      kfe_perf_stats_t *stats);

  /**
   * @brief Fused GEMM + Bias + Activation: C = activation(alpha * op(A) * op(B) + bias)
   *
   * Three-way fusion: GEMM → Bias → Activation in single pass.
   * No intermediate buffers allocated.
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
   * @param bias Bias vector [n]
   * @param activation Activation function type
   * @param C Matrix C (output) [m x n]
   * @param ldc Leading dimension of C
   * @param stats Optional performance statistics
   * @return KFE_OK on success
   */
  int kfe_sgemm_bias_activation(
      kfe_layout_t layout,
      kfe_transpose_t trans_a, kfe_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      float alpha,
      const float *A, size_t lda,
      const float *B, size_t ldb,
      const float *bias,
      kfe_activation_t activation,
      float *C, size_t ldc,
      kfe_perf_stats_t *stats);

  /**
   * @brief Fused GEMM + Element-wise Add: C = alpha * A * B + beta * D
   *
   * Element-wise addition of matrix D (not broadcasted).
   * Useful for residual connections.
   *
   * @param layout Matrix storage layout
   * @param trans_a Transpose operation for A
   * @param trans_b Transpose operation for B
   * @param m Number of rows
   * @param n Number of columns
   * @param k Inner dimension
   * @param alpha Scalar alpha
   * @param A Matrix A
   * @param lda Leading dimension of A
   * @param B Matrix B
   * @param ldb Leading dimension of B
   * @param beta Scalar beta
   * @param D Matrix D [m x n] (added element-wise)
   * @param ldd Leading dimension of D
   * @param C Matrix C (output) [m x n]
   * @param ldc Leading dimension of C
   * @param stats Optional performance statistics
   * @return KFE_OK on success
   */
  int kfe_sgemm_add(
      kfe_layout_t layout,
      kfe_transpose_t trans_a, kfe_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      float alpha,
      const float *A, size_t lda,
      const float *B, size_t ldb,
      float beta,
      const float *D, size_t ldd,
      float *C, size_t ldc,
      kfe_perf_stats_t *stats);

  /**
   * @brief Fused GEMM + Residual + Activation: C = activation(alpha*A*B + bias + beta*residual)
   *
   * Four-way fusion for ResNet-style blocks.
   *
   * @param layout Matrix storage layout
   * @param trans_a Transpose operation for A
   * @param trans_b Transpose operation for B
   * @param m Number of rows
   * @param n Number of columns
   * @param k Inner dimension
   * @param alpha Scalar alpha for GEMM
   * @param A Matrix A
   * @param lda Leading dimension of A
   * @param B Matrix B
   * @param ldb Leading dimension of B
   * @param bias Bias vector [n]
   * @param beta Scalar beta for residual
   * @param residual Residual matrix [m x n]
   * @param ldr Leading dimension of residual
   * @param activation Activation function type
   * @param C Matrix C (output) [m x n]
   * @param ldc Leading dimension of C
   * @param stats Optional performance statistics
   * @return KFE_OK on success
   */
  int kfe_sgemm_residual_activation(
      kfe_layout_t layout,
      kfe_transpose_t trans_a, kfe_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      float alpha,
      const float *A, size_t lda,
      const float *B, size_t ldb,
      const float *bias,
      float beta,
      const float *residual, size_t ldr,
      kfe_activation_t activation,
      float *C, size_t ldc,
      kfe_perf_stats_t *stats);

  /* ========================================================================== */
  /* Batched Fused Operations                                                    */
  /* ========================================================================== */

  /**
   * @brief Batched Fused GEMM + Bias + Activation
   *
   * Perform multiple fused operations in batch.
   *
   * @param layout Matrix storage layout
   * @param trans_a Transpose operation for A
   * @param trans_b Transpose operation for B
   * @param m Number of rows
   * @param n Number of columns
   * @param k Inner dimension
   * @param alpha Scalar alpha
   * @param A_array Array of matrix A pointers
   * @param lda Leading dimension of A
   * @param B_array Array of matrix B pointers
   * @param ldb Leading dimension of B
   * @param bias_array Array of bias vectors
   * @param activation Activation function type
   * @param C_array Array of matrix C pointers (output)
   * @param ldc Leading dimension of C
   * @param batch_count Number of operations in batch
   * @param stats Optional performance statistics
   * @return KFE_OK on success
   */
  int kfe_sgemm_bias_activation_batch(
      kfe_layout_t layout,
      kfe_transpose_t trans_a, kfe_transpose_t trans_b,
      size_t m, size_t n, size_t k,
      float alpha,
      const float **A_array, size_t lda,
      const float **B_array, size_t ldb,
      const float **bias_array,
      kfe_activation_t activation,
      float **C_array, size_t ldc,
      size_t batch_count,
      kfe_perf_stats_t *stats);

  /* ========================================================================== */
  /* Utility Functions                                                           */
  /* ========================================================================== */

  /**
   * @brief Estimate memory savings from fusion
   *
   * @param m Matrix dimension M
   * @param n Matrix dimension N
   * @param fusion_type Type of fusion operation
   * @return Estimated bytes saved
   */
  size_t kfe_estimate_memory_savings(size_t m, size_t n, kfe_fusion_type_t fusion_type);

  /**
   * @brief Get activation function name
   *
   * @param activation Activation type
   * @return Human-readable name
   */
  const char *kfe_activation_name(kfe_activation_t activation);

  /**
   * @brief Get fusion type name
   *
   * @param fusion_type Fusion type
   * @return Human-readable name
   */
  const char *kfe_fusion_name(kfe_fusion_type_t fusion_type);

  /**
   * @brief Convert error code to human-readable string
   *
   * @param error Error code
   * @return Error message string
   */
  const char *kfe_strerror(int error);

  /**
   * @brief Run comprehensive self-test
   *
   * Tests all fusion patterns and validates correctness.
   *
   * @param verbose Print detailed results (1/0)
   * @return KFE_OK if all tests pass
   */
  int kfe_self_test(int verbose);

  /**
   * @brief Benchmark fused vs unfused operations
   *
   * Compares performance of fused operations against traditional approach.
   *
   * @param m Matrix dimension M
   * @param n Matrix dimension N
   * @param k Matrix dimension K
   * @param iterations Number of iterations
   * @return KFE_OK on success
   */
  int kfe_benchmark_fusion(size_t m, size_t n, size_t k, int iterations);

  /**
   * @brief Get system information string
   *
   * @return Static string with CPU features, cache sizes, fusion capabilities
   */
  const char *kfe_get_system_info(void);

#ifdef __cplusplus
}
#endif

#endif /* JCORE_KERNEL_FUSION_ENGINE_H_ */