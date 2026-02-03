// advanced/jcore_neuralPrimitives/src/NPL_pooling.cpp

#include "neural_primitives_internal.h"
#include <limits>

using namespace npl_internal;

/* ========================================================================== */
/* Pooling Operations                                                          */
/* ========================================================================== */

int npl_pooling(const npl_tensor_t *input,
                 const npl_pooling_params_t *params, npl_tensor_t *output,
                 npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(input, "input");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(output, "output");
  if (ret != NPL_OK) return ret;
  ret = ValidatePoolingParams(params);
  if (ret != NPL_OK) return ret;

  auto start = std::chrono::high_resolution_clock::now();

  PoolingContext ctx;
  ret = PreparePoolingContext(input, params, output, &ctx);
  if (ret != NPL_OK) {
    return ret;
  }

  switch (params->type) {
    case NPL_POOL_MAX:
      ret = ExecuteMaxPooling(&ctx, stats);
      break;
    case NPL_POOL_AVG:
      ret = ExecuteAvgPooling(&ctx, stats);
      break;
    case NPL_POOL_GLOBAL_AVG:
      ret = ExecuteGlobalAvgPooling(input, output, stats);
      break;
    case NPL_POOL_GLOBAL_MAX:
      // Not yet implemented - fallback to max pooling
      ret = ExecuteMaxPooling(&ctx, stats);
      break;
    default:
      return NPL_ERR_UNSUPPORTED;
  }

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  if (ret == NPL_OK) {
    g_npl_state.total_ops_executed++;
  }

  return ret;
}

int npl_global_avg_pool(const npl_tensor_t *input, npl_tensor_t *output,
                         npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(input, "input");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(output, "output");
  if (ret != NPL_OK) return ret;

  auto start = std::chrono::high_resolution_clock::now();

  ret = ExecuteGlobalAvgPooling(input, output, stats);

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  if (ret == NPL_OK) {
    g_npl_state.total_ops_executed++;
  }

  return ret;
}

/* ========================================================================== */
/* Internal Implementation                                                     */
/* ========================================================================== */

namespace npl_internal {

int PreparePoolingContext(const npl_tensor_t *input,
                          const npl_pooling_params_t *params,
                          npl_tensor_t *output,
                          PoolingContext *ctx) {
  ctx->input = input;
  ctx->params = params;
  ctx->output = output;

  // Assuming NCHW layout
  if (input->ndim != 4) {
    return NPL_ERR_SHAPE_MISMATCH;
  }

  ctx->batch = input->shape[0];
  ctx->channels = input->shape[1];
  ctx->in_h = input->shape[2];
  ctx->in_w = input->shape[3];

  // Compute output dimensions
  size_t H_eff = ctx->in_h + 2 * params->pad_h;
  size_t W_eff = ctx->in_w + 2 * params->pad_w;

  ctx->out_h = (H_eff - params->kernel_h) / params->stride_h + 1;
  ctx->out_w = (W_eff - params->kernel_w) / params->stride_w + 1;

  // Verify output tensor shape
  if (output->shape[0] != ctx->batch || output->shape[1] != ctx->channels ||
      output->shape[2] != ctx->out_h || output->shape[3] != ctx->out_w) {
    return NPL_ERR_SHAPE_MISMATCH;
  }

  return NPL_OK;
}

int ExecuteMaxPooling(const PoolingContext *ctx, npl_perf_stats_t *stats) {
  const auto *p = ctx->params;
  const float *in_data = static_cast<const float *>(ctx->input->data);
  float *out_data = static_cast<float *>(ctx->output->data);

  size_t simd_width = kfe_eve_is_available() ? kfe_eve_simd_width() : 1;

  // Parallel over batch and channels
  ParallelFor(0, ctx->batch * ctx->channels, [&](size_t idx) {
    size_t n = idx / ctx->channels;
    size_t c = idx % ctx->channels;

    for (size_t oh = 0; oh < ctx->out_h; ++oh) {
      // VECTORIZED inner width loop
      size_t ow = 0;
      size_t vec_end = (ctx->out_w / simd_width) * simd_width;

      // SIMD loop over output width
      for (; ow < vec_end; ow += simd_width) {
        float max_vals[16]; // Max SIMD width buffer

        // Initialize max values
        for (size_t s = 0; s < simd_width; ++s) {
          max_vals[s] = -std::numeric_limits<float>::infinity();
        }

        for (size_t s = 0; s < simd_width && ow + s < ctx->out_w; ++s) {
          float max_val = -std::numeric_limits<float>::infinity();

          for (size_t kh = 0; kh < p->kernel_h; ++kh) {
            for (size_t kw = 0; kw < p->kernel_w; ++kw) {
              int ih = static_cast<int>(oh * p->stride_h + kh) -
                       static_cast<int>(p->pad_h);
              int iw = static_cast<int>((ow + s) * p->stride_w + kw) -
                       static_cast<int>(p->pad_w);

              if (ih >= 0 && ih < static_cast<int>(ctx->in_h) &&
                  iw >= 0 && iw < static_cast<int>(ctx->in_w)) {
                size_t in_idx = ((n * ctx->channels + c) * ctx->in_h + ih) *
                                ctx->in_w + iw;
                max_val = std::max(max_val, in_data[in_idx]);
              }
            }
          }

          max_vals[s] = max_val;
        }

        // Write vectorized results
        for (size_t s = 0; s < simd_width && ow + s < ctx->out_w; ++s) {
          size_t out_idx = ((n * ctx->channels + c) * ctx->out_h + oh) *
                           ctx->out_w + (ow + s);
          out_data[out_idx] = max_vals[s];
        }
      }

      // Scalar tail
      for (; ow < ctx->out_w; ++ow) {
        float max_val = -std::numeric_limits<float>::infinity();

        for (size_t kh = 0; kh < p->kernel_h; ++kh) {
          for (size_t kw = 0; kw < p->kernel_w; ++kw) {
            int ih = static_cast<int>(oh * p->stride_h + kh) -
                     static_cast<int>(p->pad_h);
            int iw = static_cast<int>(ow * p->stride_w + kw) -
                     static_cast<int>(p->pad_w);

            if (ih >= 0 && ih < static_cast<int>(ctx->in_h) &&
                iw >= 0 && iw < static_cast<int>(ctx->in_w)) {
              size_t in_idx = ((n * ctx->channels + c) * ctx->in_h + ih) *
                              ctx->in_w + iw;
              max_val = std::max(max_val, in_data[in_idx]);
            }
          }
        }

        size_t out_idx = ((n * ctx->channels + c) * ctx->out_h + oh) *
                         ctx->out_w + ow;
        out_data[out_idx] = max_val;
      }
    }
  });

  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "MaxPool+EVE";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

int ExecuteAvgPooling(const PoolingContext *ctx, npl_perf_stats_t *stats) {
  const auto *p = ctx->params;
  const float *in_data = static_cast<const float *>(ctx->input->data);
  float *out_data = static_cast<float *>(ctx->output->data);

  size_t simd_width = kfe_eve_is_available() ? kfe_eve_simd_width() : 1;

  ParallelFor(0, ctx->batch * ctx->channels, [&](size_t idx) {
    size_t n = idx / ctx->channels;
    size_t c = idx % ctx->channels;

    for (size_t oh = 0; oh < ctx->out_h; ++oh) {
      // VECTORIZED inner width loop
      size_t ow = 0;
      size_t vec_end = (ctx->out_w / simd_width) * simd_width;

      // SIMD loop over output width
      for (; ow < vec_end; ow += simd_width) {
        float sums[16] = {0};   // Max SIMD width buffer
        int counts[16] = {0};   // Count buffer for each SIMD lane

        for (size_t s = 0; s < simd_width && ow + s < ctx->out_w; ++s) {
          float sum = 0.0f;
          int count = 0;

          for (size_t kh = 0; kh < p->kernel_h; ++kh) {
            for (size_t kw = 0; kw < p->kernel_w; ++kw) {
              int ih = static_cast<int>(oh * p->stride_h + kh) -
                       static_cast<int>(p->pad_h);
              int iw = static_cast<int>((ow + s) * p->stride_w + kw) -
                       static_cast<int>(p->pad_w);

              if (ih >= 0 && ih < static_cast<int>(ctx->in_h) &&
                  iw >= 0 && iw < static_cast<int>(ctx->in_w)) {
                size_t in_idx = ((n * ctx->channels + c) * ctx->in_h + ih) *
                                ctx->in_w + iw;
                sum += in_data[in_idx];
                count++;
              } else if (p->count_include_pad) {
                count++;
              }
            }
          }

          sums[s] = sum;
          counts[s] = count;
        }

        // Write vectorized results
        for (size_t s = 0; s < simd_width && ow + s < ctx->out_w; ++s) {
          size_t out_idx = ((n * ctx->channels + c) * ctx->out_h + oh) *
                           ctx->out_w + (ow + s);
          out_data[out_idx] = (counts[s] > 0) ? sums[s] / counts[s] : 0.0f;
        }
      }

      // Scalar tail
      for (; ow < ctx->out_w; ++ow) {
        float sum = 0.0f;
        int count = 0;

        for (size_t kh = 0; kh < p->kernel_h; ++kh) {
          for (size_t kw = 0; kw < p->kernel_w; ++kw) {
            int ih = static_cast<int>(oh * p->stride_h + kh) -
                     static_cast<int>(p->pad_h);
            int iw = static_cast<int>(ow * p->stride_w + kw) -
                     static_cast<int>(p->pad_w);

            if (ih >= 0 && ih < static_cast<int>(ctx->in_h) &&
                iw >= 0 && iw < static_cast<int>(ctx->in_w)) {
              size_t in_idx = ((n * ctx->channels + c) * ctx->in_h + ih) *
                              ctx->in_w + iw;
              sum += in_data[in_idx];
              count++;
            } else if (p->count_include_pad) {
              count++;
            }
          }
        }

        size_t out_idx = ((n * ctx->channels + c) * ctx->out_h + oh) *
                         ctx->out_w + ow;
        out_data[out_idx] = (count > 0) ? sum / count : 0.0f;
      }
    }
  });

  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "AvgPool+EVE";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

int ExecuteGlobalAvgPooling(const npl_tensor_t *input,
                             npl_tensor_t *output,
                             npl_perf_stats_t *stats) {
  // Assuming NCHW layout
  if (input->ndim != 4) {
    return NPL_ERR_SHAPE_MISMATCH;
  }

  size_t batch = input->shape[0];
  size_t channels = input->shape[1];
  size_t height = input->shape[2];
  size_t width = input->shape[3];
  size_t spatial_size = height * width;

  const float *in_data = static_cast<const float *>(input->data);
  float *out_data = static_cast<float *>(output->data);

  size_t simd_width = kfe_eve_is_available() ? kfe_eve_simd_width() : 1;

  // VECTORIZED compute average over spatial dimensions for each channel
  ParallelFor(0, batch * channels, [&](size_t idx) {
    size_t n = idx / channels;
    size_t c = idx % channels;

    size_t offset = (n * channels + c) * spatial_size;
    const float *channel_data = in_data + offset;

    // VECTORIZED summation
    float sum = 0.0f;
    size_t vec_end = (spatial_size / simd_width) * simd_width;

    // SIMD loop - accumulate in vector register
    float simd_sums[16] = {0}; // Max SIMD width buffer
    for (size_t i = 0; i < vec_end; i += simd_width) {
      for (size_t j = 0; j < simd_width; ++j) {
        simd_sums[j] += channel_data[i + j];
      }
    }

    // Reduce SIMD partial sums
    for (size_t j = 0; j < simd_width; ++j) {
      sum += simd_sums[j];
    }

    // Scalar tail
    for (size_t i = vec_end; i < spatial_size; ++i) {
      sum += channel_data[i];
    }

    float avg = sum / spatial_size;

    // Output can be [N, C, 1, 1] or [N, C]
    if (output->ndim == 4) {
      size_t out_idx = ((n * channels + c) * 1 + 0) * 1 + 0;
      out_data[out_idx] = avg;
    } else if (output->ndim == 2) {
      size_t out_idx = n * channels + c;
      out_data[out_idx] = avg;
    }
  });

  if (stats) {
    stats->gflops = 0.0;
    stats->backend_used = "GlobalAvgPool+EVE";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

void ComputePoolingOutputShape(const size_t *input_shape,
                                const npl_pooling_params_t *params,
                                size_t *output_shape) {
  // Assuming NCHW layout
  output_shape[0] = input_shape[0]; // Batch
  output_shape[1] = input_shape[1]; // Channels

  if (params->type == NPL_POOL_GLOBAL_AVG ||
      params->type == NPL_POOL_GLOBAL_MAX) {
    output_shape[2] = 1;
    output_shape[3] = 1;
  } else {
    size_t H_eff = input_shape[2] + 2 * params->pad_h;
    size_t W_eff = input_shape[3] + 2 * params->pad_w;

    output_shape[2] = (H_eff - params->kernel_h) / params->stride_h + 1;
    output_shape[3] = (W_eff - params->kernel_w) / params->stride_w + 1;
  }
}

int ValidatePoolingParams(const npl_pooling_params_t *params) {
  if (!params) {
    return NPL_ERR_INVALID_ARG;
  }

  if (params->type != NPL_POOL_GLOBAL_AVG &&
      params->type != NPL_POOL_GLOBAL_MAX) {
    if (params->kernel_h == 0 || params->kernel_w == 0) {
      return NPL_ERR_INVALID_ARG;
    }

    if (params->stride_h == 0 || params->stride_w == 0) {
      return NPL_ERR_INVALID_ARG;
    }
  }

  return NPL_OK;
}

} // namespace NPL_internal