// advanced/neural_layer/include/neural_primitives.h
#ifndef NEURAL_PRIMITIVES_H_
#define NEURAL_PRIMITIVES_H_

/**
 * @file neural_primitives.h
 * @brief Neural Primitives Layer - High-level neural network operations
 *
 * Component: Neural Primitives Layer (Advanced)
 * Purpose: High-level math ops (Conv2D, BN, ReLU) mapped to fused kernels
 *
 * Dependencies:
 *   Derived:
 *     - Microkernel Interface Layer: GEMM/Conv operations
 *     - Vector Math Engine: Vectorized transcendental operations
 *   Advanced:
 *     - Kernel Fusion Engine: Fused operation chains
 *     - Operator Graph / Fusion Runtime: Graph-based fusion
 *     - Adaptive Graph Execution Engine: Optimized execution
 *     - JIT Kernel Generator: Runtime code generation
 *
 * Supported Operations:
 *   - Convolution: Conv2D, DepthwiseConv2D, TransposedConv2D
 *   - Activation: ReLU, ReLU6, Tanh, Sigmoid, GELU, Swish, LeakyReLU
 *   - Normalization: BatchNorm, LayerNorm, InstanceNorm, GroupNorm
 *   - Pooling: MaxPool, AvgPool, GlobalAvgPool
 *   - Tensor Ops: Add, Mul, MatMul, Concat, Split, Reshape
 *   - Fused Ops: Conv+BN+ReLU, GEMM+Bias+Activation, etc.
 *
 * Design Principles:
 *   - Automatic fusion detection and application
 *   - Memory-efficient in-place operations where possible
 *   - NUMA-aware memory allocation
 *   - Vectorized implementations for all operations
 *   - JIT-compiled kernels for uncommon shapes
 *   - Thread-safe with OpenMP/TBB parallelism
 *
 * Thread-safety: Thread-safe after initialization
 * FFM API: Fully compatible with Project JCore FFM API
 */

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Error Codes                                                                 */
/* ========================================================================== */

#define NPL_OK 0
#define NPL_ERR_NOT_INITIALIZED -1
#define NPL_ERR_INVALID_ARG -2
#define NPL_ERR_NO_MEMORY -3
#define NPL_ERR_INTERNAL -4
#define NPL_ERR_UNSUPPORTED -5
#define NPL_ERR_SHAPE_MISMATCH -6
#define NPL_ERR_OOM -7

/* ========================================================================== */
/* Data Layout                                                                 */
/* ========================================================================== */

typedef enum {
  NPL_LAYOUT_NCHW = 0,  /**< Batch, Channels, Height, Width */
  NPL_LAYOUT_NHWC = 1,  /**< Batch, Height, Width, Channels */
  NPL_LAYOUT_NCHW_VECT_C = 2  /**< Vectorized channel layout */
} npl_data_layout_t;

/* ========================================================================== */
/* Data Types                                                                  */
/* ========================================================================== */

typedef enum {
  NPL_DTYPE_FP32 = 0,   /**< 32-bit float */
  NPL_DTYPE_FP16 = 1,   /**< 16-bit float */
  NPL_DTYPE_BF16 = 2,   /**< BFloat16 */
  NPL_DTYPE_INT8 = 3,   /**< 8-bit integer (quantized) */
  NPL_DTYPE_INT32 = 4   /**< 32-bit integer */
} npl_dtype_t;

/* ========================================================================== */
/* Activation Functions                                                        */
/* ========================================================================== */

typedef enum {
  NPL_ACT_NONE = 0,
  NPL_ACT_RELU = 1,
  NPL_ACT_RELU6 = 2,
  NPL_ACT_TANH = 3,
  NPL_ACT_SIGMOID = 4,
  NPL_ACT_GELU = 5,
  NPL_ACT_SWISH = 6,
  NPL_ACT_LEAKY_RELU = 7,
  NPL_ACT_ELU = 8,
  NPL_ACT_SELU = 9,
  NPL_ACT_SOFTPLUS = 10
} npl_activation_t;

/* ========================================================================== */
/* Padding Mode                                                                */
/* ========================================================================== */

typedef enum {
  NPL_PAD_VALID = 0,    /**< No padding */
  NPL_PAD_SAME = 1,     /**< Output size = input size / stride */
  NPL_PAD_EXPLICIT = 2  /**< User-specified padding */
} npl_padding_t;

/* ========================================================================== */
/* Pooling Type                                                                */
/* ========================================================================== */

typedef enum {
  NPL_POOL_MAX = 0,
  NPL_POOL_AVG = 1,
  NPL_POOL_GLOBAL_AVG = 2,
  NPL_POOL_GLOBAL_MAX = 3
} npl_pooling_t;

/* ========================================================================== */
/* Configuration Structure                                                     */
/* ========================================================================== */

typedef struct {
  size_t num_threads;           /**< Number of threads (0 = auto) */
  int enable_fusion;            /**< Enable automatic fusion (1/0) */
  int enable_graph_optimization; /**< Use operator graph (1/0) */
  int enable_jit;               /**< Enable JIT compilation (1/0) */
  int enable_poly;               /**< Enable Polyhedral Optimization Layer (1/0) */
  int enable_vectorization;     /**< Enable SIMD vectorization (1/0) */
  int enable_numa;              /**< Enable NUMA optimization (1/0) */
  int enable_memory_pooling;    /**< Use memory pooling (1/0) */
  size_t workspace_size_mb;     /**< Workspace size in MB (0 = auto) */
  int verbose;                  /**< Verbose logging (1/0) */
} npl_config_t;

/* ========================================================================== */
/* Tensor Descriptor                                                           */
/* ========================================================================== */

#define NPL_MAX_DIMS 8

typedef struct {
  void *data;                      /**< Pointer to tensor data */
  size_t ndim;                     /**< Number of dimensions */
  size_t shape[NPL_MAX_DIMS];     /**< Dimension sizes */
  size_t strides[NPL_MAX_DIMS];   /**< Memory strides (in elements) */
  npl_dtype_t dtype;              /**< Data type */
  npl_data_layout_t layout;       /**< Data layout (for 4D tensors) */
  size_t size_bytes;               /**< Total size in bytes */
  int is_contiguous;               /**< 1 if contiguous, 0 otherwise */
} npl_tensor_t;

/* ========================================================================== */
/* Performance Statistics                                                      */
/* ========================================================================== */

typedef struct {
  double elapsed_ms;             /**< Operation time in milliseconds */
  double gflops;                 /**< Achieved GFLOPS */
  double bandwidth_gbps;         /**< Memory bandwidth in GB/s */
  size_t operations_fused;       /**< Number of operations fused */
  size_t memory_saved_bytes;     /**< Memory saved through fusion */
  const char *backend_used;      /**< Backend that executed operation */
  int was_fused;                 /**< 1 if fusion was applied */
} npl_perf_stats_t;

/* ========================================================================== */
/* Convolution Parameters                                                      */
/* ========================================================================== */

typedef struct {
  size_t kernel_h;          /**< Kernel height */
  size_t kernel_w;          /**< Kernel width */
  size_t stride_h;          /**< Vertical stride */
  size_t stride_w;          /**< Horizontal stride */
  size_t pad_h;             /**< Vertical padding */
  size_t pad_w;             /**< Horizontal padding */
  size_t dilation_h;        /**< Vertical dilation */
  size_t dilation_w;        /**< Horizontal dilation */
  size_t groups;            /**< Number of groups (1 = standard conv) */
  npl_padding_t padding;   /**< Padding mode */
  npl_activation_t activation; /**< Fused activation (NONE = no fusion) */
  int use_bias;             /**< Include bias term (1/0) */
} npl_conv_params_t;

/* ========================================================================== */
/* Batch Normalization Parameters                                             */
/* ========================================================================== */

typedef struct {
  const float *mean;        /**< Running mean [channels] */
  const float *variance;    /**< Running variance [channels] */
  const float *gamma;       /**< Scale parameter [channels] (NULL = 1.0) */
  const float *beta;        /**< Shift parameter [channels] (NULL = 0.0) */
  float epsilon;            /**< Small constant (default: 1e-5) */
  int training;             /**< Training mode (1) or inference (0) */
  float momentum;           /**< Momentum for running stats (training only) */
} npl_batchnorm_params_t;

/* ========================================================================== */
/* Layer Normalization Parameters                                             */
/* ========================================================================== */

typedef struct {
  const float *gamma;       /**< Scale parameter (NULL = 1.0) */
  const float *beta;        /**< Shift parameter (NULL = 0.0) */
  float epsilon;            /**< Small constant (default: 1e-5) */
  size_t normalized_shape;  /**< Size of normalized dimension */
} npl_layernorm_params_t;

/* ========================================================================== */
/* Pooling Parameters                                                          */
/* ========================================================================== */

typedef struct {
  size_t kernel_h;          /**< Pooling window height */
  size_t kernel_w;          /**< Pooling window width */
  size_t stride_h;          /**< Vertical stride */
  size_t stride_w;          /**< Horizontal stride */
  size_t pad_h;             /**< Vertical padding */
  size_t pad_w;             /**< Horizontal padding */
  npl_pooling_t type;      /**< Pooling type */
  int count_include_pad;    /**< Include padding in average (1/0) */
} npl_pooling_params_t;

/* ========================================================================== */
/* Initialization & Configuration                                              */
/* ========================================================================== */

/**
 * @brief Initialize Neural Primitives Layer
 *
 * Initializes all dependencies and prepares the execution environment.
 * Must be called before any NPL operations.
 *
 * @param config Configuration structure (NULL = use defaults)
 * @return NPL_OK on success, error code otherwise
 */
int npl_init(const npl_config_t *config);

/**
 * @brief Shutdown Neural Primitives Layer
 *
 * Cleanup all resources and finalize subsystems.
 */
void npl_shutdown(void);

/**
 * @brief Check if NPL is initialized
 *
 * @return 1 if initialized, 0 otherwise
 */
int npl_is_initialized(void);

/**
 * @brief Get default configuration
 *
 * @param out_config Output configuration structure
 * @return NPL_OK on success
 */
int npl_get_default_config(npl_config_t *out_config);

/* ========================================================================== */
/* Tensor Management                                                           */
/* ========================================================================== */

/**
 * @brief Create tensor descriptor
 *
 * @param data Data pointer (can be NULL if allocating later)
 * @param ndim Number of dimensions
 * @param shape Array of dimension sizes
 * @param dtype Data type
 * @param layout Data layout
 * @param out_tensor Output tensor descriptor
 * @return NPL_OK on success
 */
int npl_create_tensor(void *data, size_t ndim, const size_t *shape,
                       npl_dtype_t dtype, npl_data_layout_t layout,
                       npl_tensor_t *out_tensor);

/**
 * @brief Allocate memory for tensor
 *
 * @param tensor Tensor descriptor
 * @return NPL_OK on success
 */
int npl_allocate_tensor(npl_tensor_t *tensor);

/**
 * @brief Free tensor memory
 *
 * @param tensor Tensor descriptor
 */
void npl_free_tensor(npl_tensor_t *tensor);

/**
 * @brief Reshape tensor (view, no data copy)
 *
 * @param tensor Input tensor
 * @param new_ndim New number of dimensions
 * @param new_shape New shape
 * @param out_tensor Output reshaped tensor
 * @return NPL_OK on success
 */
int npl_reshape_tensor(const npl_tensor_t *tensor, size_t new_ndim,
                        const size_t *new_shape, npl_tensor_t *out_tensor);

/* ========================================================================== */
/* Convolution Operations                                                      */
/* ========================================================================== */

/**
 * @brief 2D Convolution
 *
 * Performs: output = Conv2D(input, weights) + bias (optional)
 *
 * @param input Input tensor [N, C_in, H, W] or [N, H, W, C_in]
 * @param weights Weight tensor [C_out, C_in/groups, K_h, K_w]
 * @param bias Bias tensor [C_out] (can be NULL)
 * @param params Convolution parameters
 * @param output Output tensor [N, C_out, H_out, W_out]
 * @param stats Optional performance statistics (NULL = don't collect)
 * @return NPL_OK on success
 */
int npl_conv2d(const npl_tensor_t *input, const npl_tensor_t *weights,
                const npl_tensor_t *bias, const npl_conv_params_t *params,
                npl_tensor_t *output, npl_perf_stats_t *stats);

/**
 * @brief Depthwise Convolution
 *
 * Efficient implementation for depthwise separable convolutions.
 *
 * @param input Input tensor [N, C, H, W]
 * @param weights Weight tensor [C, 1, K_h, K_w]
 * @param bias Bias tensor [C] (can be NULL)
 * @param params Convolution parameters
 * @param output Output tensor [N, C, H_out, W_out]
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_depthwise_conv2d(const npl_tensor_t *input,
                          const npl_tensor_t *weights,
                          const npl_tensor_t *bias,
                          const npl_conv_params_t *params,
                          npl_tensor_t *output, npl_perf_stats_t *stats);

/**
 * @brief Fused Conv2D + BatchNorm + Activation
 *
 * Three-way fusion for maximum efficiency.
 *
 * @param input Input tensor
 * @param weights Weight tensor
 * @param bias Bias tensor (can be NULL)
 * @param conv_params Convolution parameters
 * @param bn_params Batch normalization parameters
 * @param activation Activation function
 * @param output Output tensor
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_conv2d_bn_activation(const npl_tensor_t *input,
                               const npl_tensor_t *weights,
                               const npl_tensor_t *bias,
                               const npl_conv_params_t *conv_params,
                               const npl_batchnorm_params_t *bn_params,
                               npl_activation_t activation,
                               npl_tensor_t *output,
                               npl_perf_stats_t *stats);

/* ========================================================================== */
/* Activation Functions                                                        */
/* ========================================================================== */

/**
 * @brief Apply activation function
 *
 * Supports in-place operation (input == output).
 *
 * @param input Input tensor
 * @param activation Activation function type
 * @param output Output tensor (can be same as input for in-place)
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_activation(const npl_tensor_t *input, npl_activation_t activation,
                    npl_tensor_t *output, npl_perf_stats_t *stats);

/**
 * @brief ReLU activation (optimized)
 *
 * @param input Input tensor
 * @param output Output tensor (can be same as input)
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_relu(const npl_tensor_t *input, npl_tensor_t *output,
              npl_perf_stats_t *stats);

/**
 * @brief Leaky ReLU activation
 *
 * @param input Input tensor
 * @param alpha Negative slope (default: 0.01)
 * @param output Output tensor
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_leaky_relu(const npl_tensor_t *input, float alpha,
                    npl_tensor_t *output, npl_perf_stats_t *stats);

/**
 * @brief Softmax activation
 *
 * @param input Input tensor
 * @param axis Axis along which to compute softmax
 * @param output Output tensor
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_softmax(const npl_tensor_t *input, int axis, npl_tensor_t *output,
                 npl_perf_stats_t *stats);

/* ========================================================================== */
/* Normalization Operations                                                    */
/* ========================================================================== */

/**
 * @brief Batch Normalization
 *
 * @param input Input tensor [N, C, H, W]
 * @param params Batch normalization parameters
 * @param output Output tensor [N, C, H, W]
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_batch_norm(const npl_tensor_t *input,
                    const npl_batchnorm_params_t *params,
                    npl_tensor_t *output, npl_perf_stats_t *stats);

/**
 * @brief Layer Normalization
 *
 * @param input Input tensor
 * @param params Layer normalization parameters
 * @param output Output tensor
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_layer_norm(const npl_tensor_t *input,
                    const npl_layernorm_params_t *params,
                    npl_tensor_t *output, npl_perf_stats_t *stats);

/* ========================================================================== */
/* Pooling Operations                                                          */
/* ========================================================================== */

/**
 * @brief Pooling operation (Max/Avg)
 *
 * @param input Input tensor [N, C, H, W]
 * @param params Pooling parameters
 * @param output Output tensor [N, C, H_out, W_out]
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_pooling(const npl_tensor_t *input,
                 const npl_pooling_params_t *params, npl_tensor_t *output,
                 npl_perf_stats_t *stats);

/**
 * @brief Global Average Pooling
 *
 * @param input Input tensor [N, C, H, W]
 * @param output Output tensor [N, C, 1, 1] or [N, C]
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_global_avg_pool(const npl_tensor_t *input, npl_tensor_t *output,
                         npl_perf_stats_t *stats);

/* ========================================================================== */
/* Tensor Operations                                                           */
/* ========================================================================== */

/**
 * @brief Matrix multiplication
 *
 * Computes: C = alpha * A @ B + beta * C
 *
 * @param A Input tensor [M, K] or [*, M, K]
 * @param B Input tensor [K, N] or [*, K, N]
 * @param C Output tensor [M, N] or [*, M, N]
 * @param alpha Scalar multiplier for A@B
 * @param beta Scalar multiplier for C
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_matmul(const npl_tensor_t *A, const npl_tensor_t *B,
                npl_tensor_t *C, float alpha, float beta,
                npl_perf_stats_t *stats);

/**
 * @brief Element-wise addition
 *
 * Supports broadcasting. C = A + B
 *
 * @param A First input tensor
 * @param B Second input tensor
 * @param C Output tensor
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_add(const npl_tensor_t *A, const npl_tensor_t *B,
             npl_tensor_t *C, npl_perf_stats_t *stats);

/**
 * @brief Element-wise multiplication
 *
 * Supports broadcasting. C = A * B
 *
 * @param A First input tensor
 * @param B Second input tensor
 * @param C Output tensor
 * @param stats Optional performance statistics
 * @return NPL_OK on success
 */
int npl_mul(const npl_tensor_t *A, const npl_tensor_t *B,
             npl_tensor_t *C, npl_perf_stats_t *stats);

/* ========================================================================== */
/* Utility Functions                                                           */
/* ========================================================================== */

/**
 * @brief Compute output shape for convolution
 *
 * @param input_shape Input tensor shape [N, C, H, W]
 * @param params Convolution parameters
 * @param output_shape Output shape [N, C_out, H_out, W_out]
 * @return NPL_OK on success
 */
int npl_conv2d_output_shape(const size_t *input_shape,
                             const npl_conv_params_t *params,
                             size_t *output_shape);

/**
 * @brief Convert error code to string
 *
 * @param error Error code
 * @return Error message string
 */
const char *npl_strerror(int error);

/**
 * @brief Get system information
 *
 * @return Static string with CPU features, backends, capabilities
 */
const char *npl_get_system_info(void);

/**
 * @brief Run comprehensive self-test
 *
 * Tests all operations and validates correctness.
 *
 * @param verbose Print detailed results (1/0)
 * @return NPL_OK if all tests pass
 */
int npl_self_test(int verbose);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_PRIMITIVES_H_ */