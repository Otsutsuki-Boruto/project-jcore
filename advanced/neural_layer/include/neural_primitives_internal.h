// advanced/neural_layer/include/neural_primitives_internal.h
#ifndef NEURAL_PRIMITIVES_INTERNAL_H_
#define NEURAL_PRIMITIVES_INTERNAL_H_

#include "neural_primitives.h"

// Advanced component headers
#include "kernel_fusion_engine.h"
#include "kernel_fusion_eve.h"
#include "operator_graph.h"
#include "ag_execution_engine.h"
#include "jit_kernel_generator.h"

// Derived component headers
#include "microkernel_interface.h"
#include "vmath_engine.h"

// Base component headers
#include "cpu_info.h"
#include "cpu_features.h"
#include "cache_info.h"
#include "ffm_cache_block.h"
#include "thread_scheduler.h"
#include "mem_wrapper.h"
#include "ffm_prefetch.h"
#include "jcore_isa_dispatch.h"

// Standard library headers
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "global_thread_scheduler.h"

namespace npl_internal {

/* ========================================================================== */
/* Constants                                                                   */
/* ========================================================================== */

constexpr size_t DEFAULT_WORKSPACE_SIZE_MB = 256;
constexpr size_t MAX_CONV_CHANNELS = 4096;
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t MIN_PARALLEL_SIZE = 1024;
constexpr double FUSION_THRESHOLD = 1.2; // 10% speedup minimum

/* ========================================================================== */
/* Global State Structure                                                      */
/* ========================================================================== */

struct NPLState {
  bool initialized;
  npl_config_t config;

  // Component handles
  agee_session_t agee_session;
  jkg_kernel_internal_t *jit_kernels[32]; // Pre-compiled common kernels
  size_t num_jit_kernels;

  // Hardware information
  cpu_info_t cpu_info;
  CPUFeatures cpu_features;
  ffm_cache_info_t *cache_info;

  // Thread management
  jcore::global_thread::GlobalThreadScheduler *scheduler;
  size_t num_threads;

  // Memory management
  void *workspace;
  size_t workspace_size;
  std::mutex workspace_mutex;

  // Statistics
  std::atomic<size_t> total_ops_executed{0};
  std::atomic<size_t> total_ops_fused{0};
  std::atomic<size_t> total_memory_saved{0};
  std::atomic<double> total_gflops{0.0};

  // Mutex for thread-safe operations
  mutable std::mutex state_mutex;

  NPLState()
      : initialized(false), agee_session(nullptr), num_jit_kernels(0),
        cache_info(nullptr), scheduler(nullptr), num_threads(0),
        workspace(nullptr), workspace_size(0) {
    std::memset(&config, 0, sizeof(config));
    std::memset(&cpu_info, 0, sizeof(cpu_info));
    std::memset(jit_kernels, 0, sizeof(jit_kernels));
  }
};

// Global state instance (defined in npl_core.cpp)
extern NPLState g_npl_state;

/* ========================================================================== */
/* Helper Structures                                                           */
/* ========================================================================== */

/**
 * @brief Im2Col workspace for convolution
 */
struct Im2ColWorkspace {
  float *col_buffer;
  size_t col_buffer_size;
  size_t batch_size;
  size_t channels;
  size_t height;
  size_t width;
  size_t kernel_h;
  size_t kernel_w;
};

/**
 * @brief Convolution execution context
 */
struct ConvContext {
  const npl_tensor_t *input;
  const npl_tensor_t *weights;
  const npl_tensor_t *bias;
  const npl_conv_params_t *params;
  npl_tensor_t *output;

  // Computed parameters
  size_t batch;
  size_t in_channels;
  size_t out_channels;
  size_t in_h, in_w;
  size_t out_h, out_w;
  size_t pad_h_top, pad_h_bottom;
  size_t pad_w_left, pad_w_right;

  // Execution strategy
  bool use_im2col;
  bool use_winograd;
  bool use_direct;
  bool can_fuse;

  ConvContext()
      : input(nullptr), weights(nullptr), bias(nullptr), params(nullptr),
        output(nullptr), batch(0), in_channels(0), out_channels(0),
        in_h(0), in_w(0), out_h(0), out_w(0),
        pad_h_top(0), pad_h_bottom(0), pad_w_left(0), pad_w_right(0),
        use_im2col(false), use_winograd(false), use_direct(false),
        can_fuse(false) {}
};

/**
 * @brief Pooling execution context
 */
struct PoolingContext {
  const npl_tensor_t *input;
  const npl_pooling_params_t *params;
  npl_tensor_t *output;

  size_t batch;
  size_t channels;
  size_t in_h, in_w;
  size_t out_h, out_w;

  PoolingContext()
      : input(nullptr), params(nullptr), output(nullptr),
        batch(0), channels(0), in_h(0), in_w(0), out_h(0), out_w(0) {}
};

/* ========================================================================== */
/* Internal Function Declarations - Initialization                            */
/* ========================================================================== */

int InitializeComponents(const npl_config_t *config);
void ShutdownComponents();
int PrecompileCommonKernels();
void ReleasePrecompiledKernels();

/* ========================================================================== */
/* Internal Function Declarations - Convolution                               */
/* ========================================================================== */

int PrepareConvContext(const npl_tensor_t *input,
                       const npl_tensor_t *weights,
                       const npl_tensor_t *bias,
                       const npl_conv_params_t *params,
                       npl_tensor_t *output,
                       ConvContext *ctx);

int ExecuteConvDirect(const ConvContext *ctx, npl_perf_stats_t *stats);
int ExecuteConvIm2Col(const ConvContext *ctx, npl_perf_stats_t *stats);
int ExecuteConvWinograd(const ConvContext *ctx, npl_perf_stats_t *stats);
int ExecuteDepthwiseConv(const ConvContext *ctx, npl_perf_stats_t *stats);

int ExecuteConvFused(const ConvContext *ctx,
                     const npl_batchnorm_params_t *bn_params,
                     npl_activation_t activation,
                     npl_perf_stats_t *stats);

// Im2Col utilities
int AllocateIm2ColWorkspace(const ConvContext *ctx, Im2ColWorkspace *ws);
void FreeIm2ColWorkspace(Im2ColWorkspace *ws);
void Im2ColTransform(const float *input, float *col_buffer,
                     size_t channels, size_t height, size_t width,
                     size_t kernel_h, size_t kernel_w,
                     size_t stride_h, size_t stride_w,
                     size_t pad_h, size_t pad_w,
                     size_t dilation_h, size_t dilation_w);

/* ========================================================================== */
/* Internal Function Declarations - Activation                                */
/* ========================================================================== */

int ExecuteActivationScalar(const npl_tensor_t *input,
                             npl_activation_t activation,
                             npl_tensor_t *output,
                             npl_perf_stats_t *stats);

int ExecuteActivationVectorized(const npl_tensor_t *input,
                                 npl_activation_t activation,
                                 npl_tensor_t *output,
                                 npl_perf_stats_t *stats);

int ExecuteSoftmaxAxis(const npl_tensor_t *input, int axis,
                       npl_tensor_t *output, npl_perf_stats_t *stats);

// Scalar activation functions
inline float ScalarReLU(float x) { return x > 0.0f ? x : 0.0f; }
inline float ScalarReLU6(float x) { return std::min(std::max(0.0f, x), 6.0f); }
inline float ScalarLeakyReLU(float x, float alpha) {
  return x > 0.0f ? x : alpha * x;
}
inline float ScalarSigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float ScalarTanh(float x) { return std::tanh(x); }
inline float ScalarSwish(float x) { return x * ScalarSigmoid(x); }
inline float ScalarGELU(float x) {
  const float sqrt_2_over_pi = 0.7978845608f;
  const float coeff = 0.044715f;
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x3)));
}
inline float ScalarELU(float x, float alpha) {
  return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f);
}
inline float ScalarSELU(float x) {
  const float alpha = 1.67326324f;
  const float lambda = 1.05070098f;
  return lambda * (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
}
inline float ScalarSoftplus(float x) { return std::log(1.0f + std::exp(x)); }

/* ========================================================================== */
/* Internal Function Declarations - Normalization                             */
/* ========================================================================== */

int ExecuteBatchNormInference(const npl_tensor_t *input,
                               const npl_batchnorm_params_t *params,
                               npl_tensor_t *output,
                               npl_perf_stats_t *stats);

int ExecuteBatchNormTraining(const npl_tensor_t *input,
                              npl_batchnorm_params_t *params,
                              npl_tensor_t *output,
                              npl_perf_stats_t *stats);

int ExecuteLayerNormGeneric(const npl_tensor_t *input,
                             const npl_layernorm_params_t *params,
                             npl_tensor_t *output,
                             npl_perf_stats_t *stats);

// Normalization utilities
void ComputeBatchStatistics(const float *data, size_t batch, size_t channels,
                            size_t spatial_size, float *mean, float *variance);

void ApplyBatchNorm(float *data, size_t batch, size_t channels,
                    size_t spatial_size, const float *mean,
                    const float *variance, const float *gamma,
                    const float *beta, float epsilon);

/* ========================================================================== */
/* Internal Function Declarations - Pooling                                   */
/* ========================================================================== */

int PreparePoolingContext(const npl_tensor_t *input,
                          const npl_pooling_params_t *params,
                          npl_tensor_t *output,
                          PoolingContext *ctx);

int ExecuteMaxPooling(const PoolingContext *ctx, npl_perf_stats_t *stats);
int ExecuteAvgPooling(const PoolingContext *ctx, npl_perf_stats_t *stats);
int ExecuteGlobalAvgPooling(const npl_tensor_t *input,
                             npl_tensor_t *output,
                             npl_perf_stats_t *stats);

/* ========================================================================== */
/* Internal Function Declarations - Tensor Operations                         */
/* ========================================================================== */

int ExecuteMatMulDirect(const npl_tensor_t *A, const npl_tensor_t *B,
                        npl_tensor_t *C, float alpha, float beta,
                        npl_perf_stats_t *stats);

int ExecuteMatMulBatched(const npl_tensor_t *A, const npl_tensor_t *B,
                         npl_tensor_t *C, float alpha, float beta,
                         npl_perf_stats_t *stats);

int ExecuteElementwiseAdd(const npl_tensor_t *A, const npl_tensor_t *B,
                          npl_tensor_t *C, npl_perf_stats_t *stats);

int ExecuteElementwiseMul(const npl_tensor_t *A, const npl_tensor_t *B,
                          npl_tensor_t *C, npl_perf_stats_t *stats);

// Broadcasting utilities
bool CheckBroadcastable(const npl_tensor_t *A, const npl_tensor_t *B);
void ComputeBroadcastStrides(const npl_tensor_t *A, const npl_tensor_t *B,
                              size_t *strides_a, size_t *strides_b,
                              size_t *output_shape, size_t *output_ndim);

/* ========================================================================== */
/* Internal Function Declarations - Utilities                                 */
/* ========================================================================== */

// Tensor utilities
size_t GetTensorElementCount(const npl_tensor_t *tensor);
size_t GetElementSize(npl_dtype_t dtype);
bool IsTensorContiguous(const npl_tensor_t *tensor);
int MakeTensorContiguous(const npl_tensor_t *input, npl_tensor_t *output);

// Shape computation
void ComputeConvOutputShape(const size_t *input_shape,
                            const npl_conv_params_t *params,
                            size_t out_channels, size_t *output_shape);

void ComputePoolingOutputShape(const size_t *input_shape,
                                const npl_pooling_params_t *params,
                                size_t *output_shape);

// Layout conversion
int ConvertNCHWToNHWC(const npl_tensor_t *input, npl_tensor_t *output);
int ConvertNHWCToNCHW(const npl_tensor_t *input, npl_tensor_t *output);

// Memory utilities
void *AllocateAligned(size_t size, size_t alignment);
void FreeAligned(void *ptr);
void *GetWorkspacePtr(size_t required_size);
void ReleaseWorkspace();

// Performance measurement
void StartTimer(std::chrono::high_resolution_clock::time_point *start);
double GetElapsedMs(const std::chrono::high_resolution_clock::time_point &start);
double ComputeGFLOPS(double flops, double time_ms);

// Type conversion utilities
inline kfe_activation_t ToKFEActivation(npl_activation_t act) {
  switch (act) {
    case NPL_ACT_NONE: return KFE_ACTIVATION_NONE;
    case NPL_ACT_RELU: return KFE_ACTIVATION_RELU;
    case NPL_ACT_RELU6: return KFE_ACTIVATION_RELU6;
    case NPL_ACT_TANH: return KFE_ACTIVATION_TANH;
    case NPL_ACT_SIGMOID: return KFE_ACTIVATION_SIGMOID;
    case NPL_ACT_GELU: return KFE_ACTIVATION_GELU;
    case NPL_ACT_SWISH: return KFE_ACTIVATION_SWISH;
    case NPL_ACT_LEAKY_RELU: return KFE_ACTIVATION_LEAKY_RELU;
    default: return KFE_ACTIVATION_NONE;
  }
}

inline jkg_activation_t ToJKGActivation(npl_activation_t act) {
  switch (act) {
    case NPL_ACT_NONE: return JKG_ACT_NONE;
    case NPL_ACT_RELU: return JKG_ACT_RELU;
    case NPL_ACT_RELU6: return JKG_ACT_RELU6;
    case NPL_ACT_TANH: return JKG_ACT_TANH;
    case NPL_ACT_SIGMOID: return JKG_ACT_SIGMOID;
    case NPL_ACT_GELU: return JKG_ACT_GELU;
    case NPL_ACT_SWISH: return JKG_ACT_SWISH;
    case NPL_ACT_LEAKY_RELU: return JKG_ACT_LEAKY_RELU;
    default: return JKG_ACT_NONE;
  }
}

inline mil_layout_t ToMILLayout(npl_data_layout_t layout) {
  return (layout == NPL_LAYOUT_NCHW) ? MIL_LAYOUT_ROW_MAJOR
                                      : MIL_LAYOUT_COL_MAJOR;
}

/* ========================================================================== */
/* Fusion Detection and Application                                           */
/* ========================================================================== */

bool CanFuseConvBN(const ConvContext *ctx,
                   const npl_batchnorm_params_t *bn_params);

bool CanFuseActivation(const ConvContext *ctx, npl_activation_t activation);

int TryFuseOperation(const ConvContext *ctx,
                     const npl_batchnorm_params_t *bn_params,
                     npl_activation_t activation,
                     npl_perf_stats_t *stats);

/* ========================================================================== */
/* Parallel Execution Utilities                                               */
/* ========================================================================== */

struct ParallelRange {
  size_t start;
  size_t end;
  size_t thread_id;
};

template<typename Func>
void ParallelFor(size_t begin, size_t end, Func &&func) {
  if (!g_npl_state.scheduler || end - begin < MIN_PARALLEL_SIZE) {
    // Serial execution for small workloads
    for (size_t i = begin; i < end; ++i) {
      func(i);
    }
    return;
  }

  size_t num_threads = g_npl_state.num_threads;
  size_t chunk_size = (end - begin + num_threads - 1) / num_threads;

  std::vector<ParallelRange> ranges;
  for (size_t tid = 0; tid < num_threads; ++tid) {
    size_t start = begin + tid * chunk_size;
    size_t finish = std::min(start + chunk_size, end);
    if (start < finish) {
      ranges.push_back({start, finish, tid});
    }
  }

  // Execute in parallel using thread scheduler
  #pragma omp parallel for if(g_npl_state.config.num_threads > 1)
  for (size_t r = 0; r < ranges.size(); ++r) {
    for (size_t i = ranges[r].start; i < ranges[r].end; ++i) {
      func(i);
    }
  }
}

/* ========================================================================== */
/* Debugging and Validation                                                   */
/* ========================================================================== */

int ValidateTensor(const npl_tensor_t *tensor, const char *name);
int ValidateConvParams(const npl_conv_params_t *params);
int ValidatePoolingParams(const npl_pooling_params_t *params);
void PrintTensorInfo(const npl_tensor_t *tensor, const char *name);
void LogOperation(const char *op_name, const npl_perf_stats_t *stats);

} // namespace npl_internal

#endif /* NEURAL_PRIMITIVES_INTERNAL_H_ */