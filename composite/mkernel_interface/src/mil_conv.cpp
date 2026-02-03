#include "microkernel_interface.h"
#include "ffm_prefetch.h"
#include "mem_wrapper.h"

#include "pool_manager.h"
#include <mutex>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>

/* External references */
extern "C"
{
  extern int mil_is_initialized();
  extern mil_backend_t mil_get_backend();
}

namespace {
  pm_t *g_conv_pool = nullptr;
  std::mutex g_conv_pool_mutex;

  pm_t* GetConvPool() {
    std::lock_guard<std::mutex> lock(g_conv_pool_mutex);
    if (!g_conv_pool) {
      size_t pool_size = 256 * 1024 * 1024; // 256 MB for im2col buffers
      size_t chunk_size = 1 * 1024 * 1024; // 1 MB chunks
      pm_status_t status = pm_init(&g_conv_pool, pool_size, chunk_size, 0, -1);
      if (status != PM_OK) {
        g_conv_pool = nullptr;
      }
    }
    return g_conv_pool;
  }

  void CleanupConvPool() {
    std::lock_guard<std::mutex> lock(g_conv_pool_mutex);
    if (g_conv_pool) {
      pm_shutdown(g_conv_pool);
      g_conv_pool = nullptr;
    }
  }
}

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
/* Helper: Compute output dimensions                                          */
/* ========================================================================== */

static inline size_t compute_output_dim(size_t input_size, size_t kernel_size,
                                        size_t stride, size_t padding)
{
  return (input_size + 2 * padding - kernel_size) / stride + 1;
}

/* ========================================================================== */
/* Direct Convolution Implementation                                          */
/* ========================================================================== */

static void direct_conv2d_f32_impl(
    const float *input,  // [batch, in_channels, in_h, in_w]
    const float *kernel, // [out_channels, in_channels, kh, kw]
    const float *bias,   // [out_channels] or nullptr
    float *output,       // [batch, out_channels, out_h, out_w]
    size_t batch,
    size_t in_channels,
    size_t in_h, size_t in_w,
    size_t out_channels,
    size_t kh, size_t kw,
    size_t stride_h, size_t stride_w,
    size_t pad_h, size_t pad_w)
{
  size_t out_h = compute_output_dim(in_h, kh, stride_h, pad_h);
  size_t out_w = compute_output_dim(in_w, kw, stride_w, pad_w);

  // Initialize output with bias
  size_t output_size = batch * out_channels * out_h * out_w;
  if (bias != nullptr)
  {
    for (size_t b = 0; b < batch; ++b)
    {
      for (size_t oc = 0; oc < out_channels; ++oc)
      {
        for (size_t oh = 0; oh < out_h; ++oh)
        {
          for (size_t ow = 0; ow < out_w; ++ow)
          {
            size_t out_idx = b * (out_channels * out_h * out_w) +
                             oc * (out_h * out_w) +
                             oh * out_w + ow;
            output[out_idx] = bias[oc];
          }
        }
      }
    }
  }
  else
  {
    std::memset(output, 0, output_size * sizeof(float));
  }

  // Perform convolution
  for (size_t b = 0; b < batch; ++b)
  {
    for (size_t oc = 0; oc < out_channels; ++oc)
    {
      for (size_t ic = 0; ic < in_channels; ++ic)
      {
        for (size_t oh = 0; oh < out_h; ++oh)
        {
          for (size_t ow = 0; ow < out_w; ++ow)
          {

            // Compute input position (accounting for stride and padding)
            size_t in_h_start = oh * stride_h;
            size_t in_w_start = ow * stride_w;

            float sum = 0.0f;

            // Kernel loop
            for (size_t kh_idx = 0; kh_idx < kh; ++kh_idx)
            {
              for (size_t kw_idx = 0; kw_idx < kw; ++kw_idx)
              {

                // Check bounds (handle padding)
                long in_h_pos = static_cast<long>(in_h_start) + static_cast<long>(kh_idx) - static_cast<long>(pad_h);
                long in_w_pos = static_cast<long>(in_w_start) + static_cast<long>(kw_idx) - static_cast<long>(pad_w);

                if (in_h_pos >= 0 && in_h_pos < static_cast<long>(in_h) &&
                    in_w_pos >= 0 && in_w_pos < static_cast<long>(in_w))
                {

                  size_t in_idx = b * (in_channels * in_h * in_w) +
                                  ic * (in_h * in_w) +
                                  in_h_pos * in_w + in_w_pos;

                  size_t k_idx = oc * (in_channels * kh * kw) +
                                 ic * (kh * kw) +
                                 kh_idx * kw + kw_idx;

                  // Prefetch next input element in kernel window
                  if (kw_idx + 1 < kw)
                  {
                    long next_w_pos = static_cast<long>(in_w_start) + static_cast<long>(kw_idx + 1) - static_cast<long>(pad_w);
                    if (in_h_pos >= 0 && in_h_pos < static_cast<long>(in_h) &&
                        next_w_pos >= 0 && next_w_pos < static_cast<long>(in_w))
                    {
                      size_t next_in_idx = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + in_h_pos * in_w + next_w_pos;
                      ffm_prefetch_addr_read(&input[next_in_idx]);
                    }
                  }

                  sum += input[in_idx] * kernel[k_idx];
                }
              }
            }

            size_t out_idx = b * (out_channels * out_h * out_w) +
                             oc * (out_h * out_w) +
                             oh * out_w + ow;
            output[out_idx] += sum;
          }
        }
      }
    }
  }
}

/* ========================================================================== */
/* Im2Col Helper Function                                                     */
/* ========================================================================== */

static float *im2col_transform(
    const float *input,
    size_t batch_idx,
    size_t in_channels,
    size_t in_h, size_t in_w,
    size_t kh, size_t kw,
    size_t stride_h, size_t stride_w,
    size_t pad_h, size_t pad_w,
    size_t out_h, size_t out_w)
{
  // Allocate im2col buffer: [in_channels * kh * kw, out_h * out_w]
  size_t col_h = in_channels * kh * kw;
  size_t col_w = out_h * out_w;

  pm_t *pool = GetConvPool();
  float *col = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(ffm_malloc(col_h * col_w * sizeof(float)));
  if (col == nullptr)
  {
    return nullptr;
  }

  // Transform input to column format
  size_t col_idx = 0;
  for (size_t ic = 0; ic < in_channels; ++ic)
  {
    for (size_t kh_idx = 0; kh_idx < kh; ++kh_idx)
    {
      for (size_t kw_idx = 0; kw_idx < kw; ++kw_idx)
      {
        for (size_t oh = 0; oh < out_h; ++oh)
        {
          for (size_t ow = 0; ow < out_w; ++ow)
          {

            long in_h_pos = static_cast<long>(oh * stride_h) + static_cast<long>(kh_idx) - static_cast<long>(pad_h);
            long in_w_pos = static_cast<long>(ow * stride_w) + static_cast<long>(kw_idx) - static_cast<long>(pad_w);

            if (in_h_pos >= 0 && in_h_pos < static_cast<long>(in_h) &&
                in_w_pos >= 0 && in_w_pos < static_cast<long>(in_w))
            {
              size_t in_idx = batch_idx * (in_channels * in_h * in_w) +
                              ic * (in_h * in_w) +
                              in_h_pos * in_w + in_w_pos;
              col[col_idx] = input[in_idx];
            }
            else
            {
              col[col_idx] = 0.0f; // Padding
            }
            col_idx++;
          }
        }
      }
    }
  }

  return col;
}

/* ========================================================================== */
/* Im2Col + GEMM Convolution Implementation                                   */
/* ========================================================================== */

static int im2col_conv2d_f32_impl(
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
    size_t pad_h, size_t pad_w)
{
  size_t out_h = compute_output_dim(in_h, kh, stride_h, pad_h);
  size_t out_w = compute_output_dim(in_w, kw, stride_w, pad_w);

  // Reshape kernel to [out_channels, in_channels * kh * kw]
  size_t kernel_rows = out_channels;
  size_t kernel_cols = in_channels * kh * kw;

  pm_t *pool = GetConvPool();

  // Process each batch independently
  for (size_t b = 0; b < batch; ++b)
  {
    // Transform input to column format
    float *col = im2col_transform(
        input, b, in_channels, in_h, in_w,
        kh, kw, stride_h, stride_w, pad_h, pad_w,
        out_h, out_w);

    if (col == nullptr)
    {
      return MIL_ERR_ALLOCATION;
    }

    // Now perform matrix multiply: kernel * col = output_slice
    // kernel: [out_channels, kernel_cols]
    // col:    [kernel_cols, out_h * out_w]
    // output: [out_channels, out_h * out_w]

    size_t output_slice_size = out_channels * out_h * out_w;
    float *output_slice = output + b * output_slice_size;

    // Initialize with bias
    if (bias != nullptr)
    {
      for (size_t oc = 0; oc < out_channels; ++oc)
      {
        for (size_t i = 0; i < out_h * out_w; ++i)
        {
          output_slice[oc * out_h * out_w + i] = bias[oc];
        }
      }
    }
    else
    {
      std::memset(output_slice, 0, output_slice_size * sizeof(float));
    }

    // Matrix multiply (simple implementation - could call mil_sgemm here)
    for (size_t oc = 0; oc < out_channels; ++oc)
    {
      for (size_t spatial = 0; spatial < out_h * out_w; ++spatial)
      {
        float sum = 0.0f;
        for (size_t k = 0; k < kernel_cols; ++k)
        {
          sum += kernel[oc * kernel_cols + k] * col[k * (out_h * out_w) + spatial];
        }
        output_slice[oc * out_h * out_w + spatial] += sum;
      }
    }

    // Free im2col buffer
    if (pool) {
      pm_free(pool, col);
    } else {
      ffm_free(col);
    }
  }

  return MIL_OK;
}

/* ========================================================================== */
/* Public API: Direct Convolution                                             */
/* ========================================================================== */

extern "C"
{

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
      mil_perf_stats_t *stats)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (input == nullptr || kernel == nullptr || output == nullptr)
    {
      return MIL_ERR_INVALID_ARG;
    }

    if (batch == 0 || in_channels == 0 || out_channels == 0 ||
        in_h == 0 || in_w == 0 || kh == 0 || kw == 0)
    {
      return MIL_ERR_INVALID_ARG;
    }

    double start_time = get_time_ms();

    direct_conv2d_f32_impl(
        input, kernel, bias, output,
        batch, in_channels, in_h, in_w,
        out_channels, kh, kw,
        stride_h, stride_w, pad_h, pad_w);

    double elapsed_ms = get_time_ms() - start_time;

    if (stats != nullptr)
    {
      size_t out_h = compute_output_dim(in_h, kh, stride_h, pad_h);
      size_t out_w = compute_output_dim(in_w, kw, stride_w, pad_w);

      // Operations: 2 * batch * out_channels * in_channels * kh * kw * out_h * out_w
      double ops = 2.0 * static_cast<double>(batch) * static_cast<double>(out_channels) *
                   static_cast<double>(in_channels) * static_cast<double>(kh) *
                   static_cast<double>(kw) * static_cast<double>(out_h) * static_cast<double>(out_w);

      stats->gflops = (ops / 1e9) / (elapsed_ms / 1000.0);
      stats->elapsed_ms = elapsed_ms;

      size_t input_bytes = batch * in_channels * in_h * in_w * sizeof(float);
      size_t kernel_bytes = out_channels * in_channels * kh * kw * sizeof(float);
      size_t output_bytes = batch * out_channels * out_h * out_w * sizeof(float);
      stats->bytes_transferred = input_bytes + kernel_bytes + output_bytes;
      stats->bandwidth_gbps = (stats->bytes_transferred / 1e9) / (elapsed_ms / 1000.0);

      stats->kernel_used = "direct_conv2d";
      stats->backend_used = mil_get_backend();
    }

    return MIL_OK;
  }

  /* ========================================================================== */
  /* Public API: Im2Col + GEMM Convolution                                      */
  /* ========================================================================== */

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
      mil_perf_stats_t *stats)
  {
    if (!mil_is_initialized())
    {
      return MIL_ERR_NOT_INITIALIZED;
    }

    if (input == nullptr || kernel == nullptr || output == nullptr)
    {
      return MIL_ERR_INVALID_ARG;
    }

    if (batch == 0 || in_channels == 0 || out_channels == 0 ||
        in_h == 0 || in_w == 0 || kh == 0 || kw == 0)
    {
      return MIL_ERR_INVALID_ARG;
    }

    double start_time = get_time_ms();

    int status = im2col_conv2d_f32_impl(
        input, kernel, bias, output,
        batch, in_channels, in_h, in_w,
        out_channels, kh, kw,
        stride_h, stride_w, pad_h, pad_w);

    if (status != MIL_OK)
    {
      return status;
    }

    double elapsed_ms = get_time_ms() - start_time;

    if (stats != nullptr)
    {
      size_t out_h = compute_output_dim(in_h, kh, stride_h, pad_h);
      size_t out_w = compute_output_dim(in_w, kw, stride_w, pad_w);

      double ops = 2.0 * static_cast<double>(batch) * static_cast<double>(out_channels) *
                   static_cast<double>(in_channels) * static_cast<double>(kh) *
                   static_cast<double>(kw) * static_cast<double>(out_h) * static_cast<double>(out_w);

      stats->gflops = (ops / 1e9) / (elapsed_ms / 1000.0);
      stats->elapsed_ms = elapsed_ms;

      size_t input_bytes = batch * in_channels * in_h * in_w * sizeof(float);
      size_t kernel_bytes = out_channels * in_channels * kh * kw * sizeof(float);
      size_t output_bytes = batch * out_channels * out_h * out_w * sizeof(float);
      stats->bytes_transferred = input_bytes + kernel_bytes + output_bytes;
      stats->bandwidth_gbps = (stats->bytes_transferred / 1e9) / (elapsed_ms / 1000.0);

      stats->kernel_used = "im2col_conv2d";
      stats->backend_used = mil_get_backend();
    }

    return MIL_OK;
  }

} // extern "C"