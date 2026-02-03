// advanced/jcore_neuralPrimitives/src/NPL_activation_norm.cpp

#include "ffm_prefetch.h"
#include "neural_primitives.h"
#include "neural_primitives_internal.h"

using namespace npl_internal;

/* ========================================================================== */
/* Activation Functions                                                        */
/* ========================================================================== */

int npl_activation(const npl_tensor_t *input, npl_activation_t activation,
                    npl_tensor_t *output, npl_perf_stats_t *stats) {

  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(input, "input");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(output, "output");
  if (ret != NPL_OK) return ret;

  if (activation == NPL_ACT_NONE) {
    // No-op or copy
    if (input->data != output->data) {
      std::memcpy(output->data, input->data, input->size_bytes);
    }
    return NPL_OK;
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Use vectorized implementation if available
  if (g_npl_state.config.enable_vectorization) {
    ret = ExecuteActivationVectorized(input, activation, output, stats);
  } else {
    ret = ExecuteActivationScalar(input, activation, output, stats);
  }

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  g_npl_state.total_ops_executed++;

  return ret;
}

int npl_relu(const npl_tensor_t *input, npl_tensor_t *output,
              npl_perf_stats_t *stats) {
  return npl_activation(input, NPL_ACT_RELU, output, stats);
}

int npl_leaky_relu(const npl_tensor_t *input, float alpha,
                    npl_tensor_t *output, npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(input, "input");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(output, "output");
  if (ret != NPL_OK) return ret;

  auto start = std::chrono::high_resolution_clock::now();

  size_t n = GetTensorElementCount(input);
  const float *in_data = static_cast<const float *>(input->data);
  float *out_data = static_cast<float *>(output->data);

  ParallelFor(0, n, [&](size_t i) {
    out_data[i] = ScalarLeakyReLU(in_data[i], alpha);
  });

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
    stats->gflops = 0.0;
    stats->backend_used = "Scalar";
    stats->was_fused = 0;
  }

  g_npl_state.total_ops_executed++;
  return NPL_OK;
}

int npl_softmax(const npl_tensor_t *input, int axis, npl_tensor_t *output,
                 npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(input, "input");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(output, "output");
  if (ret != NPL_OK) return ret;

  auto start = std::chrono::high_resolution_clock::now();

  ret = ExecuteSoftmaxAxis(input, axis, output, stats);

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  if (ret == NPL_OK) {
    g_npl_state.total_ops_executed++;
  }

  return ret;
}

/* ========================================================================== */
/* Normalization Operations                                                    */
/* ========================================================================== */

int npl_batch_norm(const npl_tensor_t *input,
                    const npl_batchnorm_params_t *params,
                    npl_tensor_t *output, npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(input, "input");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(output, "output");
  if (ret != NPL_OK) return ret;

  if (!params || !params->mean || !params->variance) {
    return NPL_ERR_INVALID_ARG;
  }

  auto start = std::chrono::high_resolution_clock::now();

  if (params->training) {
    ret = ExecuteBatchNormTraining(input,
                                   const_cast<npl_batchnorm_params_t*>(params),
                                   output, stats);
  } else {
    ret = ExecuteBatchNormInference(input, params, output, stats);
  }

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  if (ret == NPL_OK) {
    g_npl_state.total_ops_executed++;
  }

  return ret;
}

int npl_layer_norm(const npl_tensor_t *input,
                    const npl_layernorm_params_t *params,
                    npl_tensor_t *output, npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(input, "input");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(output, "output");
  if (ret != NPL_OK) return ret;

  if (!params) {
    return NPL_ERR_INVALID_ARG;
  }

  auto start = std::chrono::high_resolution_clock::now();

  ret = ExecuteLayerNormGeneric(input, params, output, stats);

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  g_npl_state.total_ops_executed++;

  return ret;
}

/* ========================================================================== */
/* Internal Implementation - Activation                                        */
/* ========================================================================== */

namespace npl_internal {

int ExecuteActivationScalar(const npl_tensor_t *input,
                               npl_activation_t activation,
                               npl_tensor_t *output,
                               npl_perf_stats_t *stats) {
    size_t n = GetTensorElementCount(input);
    const float *in_data = static_cast<const float *>(input->data);
    float *out_data = static_cast<float *>(output->data);

    // Prefetch input/output tensors
    ffm_prefetch_block_read_T0(input->data, n * sizeof(float));
    ffm_prefetch_block_write_T0(output->data, n * sizeof(float));

    ParallelFor(0, n, [&](size_t i) {
    float x = in_data[i];
    float y = 0.0f;

    switch (activation) {
      case NPL_ACT_RELU: y = ScalarReLU(x); break;
      case NPL_ACT_RELU6: y = ScalarReLU6(x); break;
      case NPL_ACT_TANH: y = ScalarTanh(x); break;
      case NPL_ACT_SIGMOID: y = ScalarSigmoid(x); break;
      case NPL_ACT_GELU: y = ScalarGELU(x); break;
      case NPL_ACT_SWISH: y = ScalarSwish(x); break;
      case NPL_ACT_LEAKY_RELU: y = ScalarLeakyReLU(x, 0.01f); break;
      case NPL_ACT_ELU: y = ScalarELU(x, 1.0f); break;
      case NPL_ACT_SELU: y = ScalarSELU(x); break;
      case NPL_ACT_SOFTPLUS: y = ScalarSoftplus(x); break;
      default: y = x; break;
    }

    out_data[i] = y;
  });

  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "Scalar";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

int ExecuteActivationVectorized(const npl_tensor_t *input,
                                   npl_activation_t activation,
                                   npl_tensor_t *output,
                                   npl_perf_stats_t *stats) {
  size_t n = GetTensorElementCount(input);
  float *data = static_cast<float *>(output->data);

  // Prefetch input/output tensors
  ffm_prefetch_block_read_T0(input->data, n * sizeof(float));
  ffm_prefetch_block_write_T0(output->data, n * sizeof(float));

  // Copy input to output if different
  if (input->data != output->data) {
    std::memcpy(output->data, input->data, input->size_bytes);
  }

  // Use Microkernel Interface for transcendental functions
  int ret = VMATH_OK;
  switch (activation) {
    case NPL_ACT_TANH:
      printf("[INFO] CASE : NPL_ACT_TANH\n");
      ret = mil_tanhf(data, data, n);
      break;
    case NPL_ACT_SIGMOID:
      printf("[INFO] CASE : NPL_ACT_SIGMOID\n");
      ret = mil_sigmoidf(data, data, n);
      break;
    case NPL_ACT_GELU:
      printf("[INFO] CASE : NPL_ACT_GELU\n");
      ret = mil_geluf(data, data, n);
      break;
    case NPL_ACT_SOFTPLUS:
      printf("[INFO] CASE : NPL_ACT_SOFTPLUS\n");
      ret = mil_softplusf(data, data, n);
      break;
    case NPL_ACT_RELU:
      printf("[INFO] CASE : NPL_ACT_RELU\n");
      ret = mil_reluf(data, data, n);
      break;
    case NPL_ACT_RELU6:
      printf("[INFO] CASE : NPL_ACT_RELU6\n");
      ret = mil_relu6f(data, data, n);
      break;
    default:
      // Fallback to scalar for other activations
      printf("[INFO] FallBack to Scalar Activations\n");
      return ExecuteActivationScalar(input, activation, output, stats);
  }

  if (ret != VMATH_OK) {
    return NPL_ERR_INTERNAL;
  }

  printf("[INFO] ExecutionVectorized Completed\n");
  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "VMath";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

int ExecuteSoftmaxAxis(const npl_tensor_t *input, int axis,
                       npl_tensor_t *output, npl_perf_stats_t *stats) {
  // Handle negative axis
  if (axis < 0) {
    axis = static_cast<int>(input->ndim) + axis;
  }

  if (axis < 0 || axis >= static_cast<int>(input->ndim)) {
    return NPL_ERR_INVALID_ARG;
  }

  const float *in_data = static_cast<const float *>(input->data);
  float *out_data = static_cast<float *>(output->data);

  size_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input->shape[i];
  }

  size_t axis_size = input->shape[axis];

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < input->ndim; ++i) {
    inner_size *= input->shape[i];
  }

  // Softmax computation
  ParallelFor(0, outer_size * inner_size, [&](size_t idx) {
    size_t outer = idx / inner_size;
    size_t inner = idx % inner_size;

    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < axis_size; ++i) {
      size_t offset = (outer * axis_size + i) * inner_size + inner;
      max_val = std::max(max_val, in_data[offset]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < axis_size; ++i) {
      size_t offset = (outer * axis_size + i) * inner_size + inner;
      float exp_val = std::exp(in_data[offset] - max_val);
      out_data[offset] = exp_val;
      sum += exp_val;
    }

    // Normalize
    for (size_t i = 0; i < axis_size; ++i) {
      size_t offset = (outer * axis_size + i) * inner_size + inner;
      out_data[offset] /= sum;
    }
  });

  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "Softmax";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

/* ========================================================================== */
/* Internal Implementation - Normalization                                     */
/* ========================================================================== */

int ExecuteBatchNormInference(const npl_tensor_t *input,
                               const npl_batchnorm_params_t *params,
                               npl_tensor_t *output,
                               npl_perf_stats_t *stats) {
  // Assuming NCHW layout: [batch, channels, height, width]
  if (input->ndim != 4) {
    return NPL_ERR_SHAPE_MISMATCH;
  }

  size_t batch = input->shape[0];
  size_t channels = input->shape[1];
  size_t spatial_size = input->shape[2] * input->shape[3];

  const float *in_data = static_cast<const float *>(input->data);

  // Prefetch input/output tensors
  ffm_prefetch_block_read_T0(input->data, batch * channels * spatial_size * sizeof(float));
  ffm_prefetch_block_write_T0(output->data, batch * channels * spatial_size * sizeof(float));
  float *out_data = static_cast<float *>(output->data);

  // Apply normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
  ParallelFor(0, batch * channels, [&](size_t idx) {
    size_t n = idx / channels;
    size_t c = idx % channels;

    float mean = params->mean[c];
    float var = params->variance[c];
    float gamma = params->gamma ? params->gamma[c] : 1.0f;
    float beta = params->beta ? params->beta[c] : 0.0f;
    float inv_std = 1.0f / std::sqrt(var + params->epsilon);

    size_t offset = (n * channels + c) * spatial_size;
    for (size_t i = 0; i < spatial_size; ++i) {
      float x = in_data[offset + i];
      out_data[offset + i] = gamma * (x - mean) * inv_std + beta;
    }
  });

  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "BatchNorm";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

int ExecuteBatchNormTraining(const npl_tensor_t *input,
                              npl_batchnorm_params_t *params,
                              npl_tensor_t *output,
                              npl_perf_stats_t *stats) {
  // Training mode: compute batch statistics
  if (input->ndim != 4) {
    return NPL_ERR_SHAPE_MISMATCH;
  }

  size_t batch = input->shape[0];
  size_t channels = input->shape[1];
  size_t spatial_size = input->shape[2] * input->shape[3];

  const float *in_data = static_cast<const float *>(input->data);
  float *out_data = static_cast<float *>(output->data);

  // Compute batch mean and variance
  std::vector<float> batch_mean(channels, 0.0f);
  std::vector<float> batch_var(channels, 0.0f);

  ComputeBatchStatistics(in_data, batch, channels, spatial_size,
                         batch_mean.data(), batch_var.data());

  // Update running statistics (need to modify mean/variance)
  // Cast away const since we're in training mode and need to update running stats
  float *mean_mut = const_cast<float*>(params->mean);
  float *var_mut = const_cast<float*>(params->variance);

  float momentum = params->momentum;
  for (size_t c = 0; c < channels; ++c) {
    mean_mut[c] = momentum * mean_mut[c] + (1.0f - momentum) * batch_mean[c];
    var_mut[c] = momentum * var_mut[c] + (1.0f - momentum) * batch_var[c];
  }

  // Apply normalization using batch statistics
  ParallelFor(0, batch * channels, [&](size_t idx) {
    size_t n = idx / channels;
    size_t c = idx % channels;

    float mean = batch_mean[c];
    float var = batch_var[c];
    float gamma = params->gamma ? params->gamma[c] : 1.0f;
    float beta = params->beta ? params->beta[c] : 0.0f;
    float inv_std = 1.0f / std::sqrt(var + params->epsilon);

    size_t offset = (n * channels + c) * spatial_size;
    for (size_t i = 0; i < spatial_size; ++i) {
      float x = in_data[offset + i];
      out_data[offset + i] = gamma * (x - mean) * inv_std + beta;
    }
  });

  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "BatchNorm-Train";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

int ExecuteLayerNormGeneric(const npl_tensor_t *input,
                             const npl_layernorm_params_t *params,
                             npl_tensor_t *output,
                             npl_perf_stats_t *stats) {
  // Layer norm: normalize over last dimension
  size_t normalized_shape = params->normalized_shape;
  size_t outer_size = GetTensorElementCount(input) / normalized_shape;

  const float *in_data = static_cast<const float *>(input->data);
  float *out_data = static_cast<float *>(output->data);

  // Prefetch input/output tensors
  size_t total_elements = outer_size * normalized_shape;
  ffm_prefetch_block_read_T0(input->data, total_elements * sizeof(float));
  ffm_prefetch_block_write_T0(output->data, total_elements * sizeof(float));

  ParallelFor(0, outer_size, [&](size_t i) {
    size_t offset = i * normalized_shape;

    // Compute mean
    float mean = 0.0f;
    for (size_t j = 0; j < normalized_shape; ++j) {
      mean += in_data[offset + j];
    }
    mean /= normalized_shape;

    // Compute variance
    float var = 0.0f;
    for (size_t j = 0; j < normalized_shape; ++j) {
      float diff = in_data[offset + j] - mean;
      var += diff * diff;
    }
    var /= normalized_shape;

    // Normalize
    float inv_std = 1.0f / std::sqrt(var + params->epsilon);
    for (size_t j = 0; j < normalized_shape; ++j) {
      float x = in_data[offset + j];
      float norm_x = (x - mean) * inv_std;
      float gamma = params->gamma ? params->gamma[j] : 1.0f;
      float beta = params->beta ? params->beta[j] : 0.0f;
      out_data[offset + j] = gamma * norm_x + beta;
    }
  });

  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "LayerNorm";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

void ComputeBatchStatistics(const float *data, size_t batch, size_t channels,
                            size_t spatial_size, float *mean, float *variance) {
  size_t N = batch * spatial_size;

  // Compute mean per channel
  for (size_t c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (size_t n = 0; n < batch; ++n) {
      size_t offset = (n * channels + c) * spatial_size;
      for (size_t i = 0; i < spatial_size; ++i) {
        sum += data[offset + i];
      }
    }
    mean[c] = sum / N;
  }

  // Compute variance per channel
  for (size_t c = 0; c < channels; ++c) {
    float sum_sq = 0.0f;
    for (size_t n = 0; n < batch; ++n) {
      size_t offset = (n * channels + c) * spatial_size;
      for (size_t i = 0; i < spatial_size; ++i) {
        float diff = data[offset + i] - mean[c];
        sum_sq += diff * diff;
      }
    }
    variance[c] = sum_sq / N;
  }
}

void ApplyBatchNorm(float *data, size_t batch, size_t channels,
                    size_t spatial_size, const float *mean,
                    const float *variance, const float *gamma,
                    const float *beta, float epsilon) {
  ParallelFor(0, batch * channels, [&](size_t idx) {
    size_t n = idx / channels;
    size_t c = idx % channels;

    float m = mean[c];
    float v = variance[c];
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;
    float inv_std = 1.0f / std::sqrt(v + epsilon);

    size_t offset = (n * channels + c) * spatial_size;
    for (size_t i = 0; i < spatial_size; ++i) {
      float x = data[offset + i];
      data[offset + i] = g * (x - m) * inv_std + b;
    }
  });
}

} // namespace NPL_internal