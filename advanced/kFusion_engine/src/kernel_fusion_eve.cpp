// advanced/kFusion_engine/src/kernel_fusion_eve.cpp

#include "kernel_fusion_eve.h"
#include "kernel_fusion_engine_internal.h"

// EVE library includes
#include <eve/eve.hpp>
#include <eve/module/core/regular/reduce.hpp>
#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

using namespace kfe_internal;

namespace {

/* ========================================================================== */
/* EVE-Powered Activation Functions                                           */
/* ========================================================================== */

// EVE supports any SIMD width automatically
template <typename Wide> EVE_FORCEINLINE Wide eve_relu(Wide x) {
  return eve::max(x, Wide{0.0f});
}

template <typename Wide> EVE_FORCEINLINE Wide eve_relu6(Wide x) {
  return eve::min(eve::max(x, Wide{0.0f}), Wide{6.0f});
}

template <typename Wide> EVE_FORCEINLINE Wide eve_leaky_relu(Wide x) {
  return eve::if_else(x > 0.0f, x, x * 0.01f);
}

template <typename Wide> EVE_FORCEINLINE Wide eve_sigmoid(Wide x) {
  // sigmoid(x) = 1 / (1 + exp(-x))
  // Clamp for numerical stability
  auto clamped = eve::clamp(x, Wide{-10.0f}, Wide{10.0f});
  return eve::rec(Wide{1.0f} + eve::exp(-clamped));
}

template <typename Wide> EVE_FORCEINLINE Wide eve_tanh_activation(Wide x) {
  // Use EVE's optimized tanh
  return eve::tanh(x);
}

template <typename Wide> EVE_FORCEINLINE Wide eve_gelu(Wide x) {
  // GELU approximation: 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))
  constexpr float sqrt_2_over_pi = 0.7978845608f;
  constexpr float coeff = 0.044715f;

  auto x3 = x * x * x;
  auto inner = Wide{sqrt_2_over_pi} * (x + Wide{coeff} * x3);
  return Wide{0.5f} * x * (Wide{1.0f} + eve::tanh(inner));
}

template <typename Wide> EVE_FORCEINLINE Wide eve_swish(Wide x) {
  // Swish: x * sigmoid(x)
  return x * eve_sigmoid(x);
}

// Generic activation dispatcher
template <typename Wide>
EVE_FORCEINLINE Wide apply_eve_activation(Wide x, kfe_activation_t act) {
  switch (act) {
  case KFE_ACTIVATION_RELU:
    return eve_relu(x);
  case KFE_ACTIVATION_RELU6:
    return eve_relu6(x);
  case KFE_ACTIVATION_LEAKY_RELU:
    return eve_leaky_relu(x);
  case KFE_ACTIVATION_SIGMOID:
    return eve_sigmoid(x);
  case KFE_ACTIVATION_TANH:
    return eve_tanh_activation(x);
  case KFE_ACTIVATION_GELU:
    return eve_gelu(x);
  case KFE_ACTIVATION_SWISH:
    return eve_swish(x);
  default:
    return x; // NONE
  }
}

/* ========================================================================== */
/* EVE-Powered Fused Epilogue Pipeline                                        */
/* ========================================================================== */

// Process single row with full epilogue fusion
static void process_row_eve_epilogue(float *C_row, size_t n, const float *bias,
                                     const kfe_batchnorm_params_t *bn_params,
                                     const float *residual_row,
                                     const kfe_epilogue_config_t *config) {
  using wide_t = eve::wide<float>;
  constexpr size_t simd_width = wide_t::size();

  size_t i = 0;
  const size_t vec_end = (n / simd_width) * simd_width;

  // Vectorized main loop
  for (; i < vec_end; i += simd_width) {
    // Load
    wide_t x = wide_t{&C_row[i]};

    // 1. Bias addition
    if (config->enable_bias && bias) {
      wide_t b = wide_t{&bias[i]};
      x = x + b;
    }

    // 2. Batch normalization (vectorized)
    if (config->enable_normalization && bn_params &&
        config->norm_type == KFE_NORM_BATCH) {
      // Load BN parameters as vectors
      wide_t mean_vec{&bn_params->mean[i]};
      wide_t var_vec{&bn_params->variance[i]};
      wide_t gamma_vec{&bn_params->gamma[i]};
      wide_t beta_vec{&bn_params->beta[i]};
      wide_t eps_vec{bn_params->epsilon};

      // Vectorized BN: (x - mean) / sqrt(var + eps) * gamma + beta
      x = (x - mean_vec) * eve::rsqrt(var_vec + eps_vec) * gamma_vec + beta_vec;
    }

    // 3. Residual connection
    if (config->enable_residual && residual_row) {
      wide_t res = wide_t{&residual_row[i]};
      x = x + res;
    }

    // 4. Activation
    if (config->enable_activation) {
      x = apply_eve_activation(x, config->activation);
    }

    // Store
    eve::store(x, &C_row[i]);
  }

  // Scalar tail
  for (; i < n; ++i) {
    float x = C_row[i];

    if (config->enable_bias && bias) {
      x += bias[i];
    }

    if (config->enable_normalization && bn_params &&
        config->norm_type == KFE_NORM_BATCH) {
      x = (x - bn_params->mean[i]) *
              (1.0f / std::sqrt(bn_params->variance[i] + bn_params->epsilon)) *
              bn_params->gamma[i] +
          bn_params->beta[i];
    }

    if (config->enable_residual && residual_row) {
      x += residual_row[i];
    }

    if (config->enable_activation) {
      switch (config->activation) {
      case KFE_ACTIVATION_RELU:
        x = relu(x);
        break;
      case KFE_ACTIVATION_RELU6:
        x = relu6(x);
        break;
      case KFE_ACTIVATION_LEAKY_RELU:
        x = leaky_relu(x);
        break;
      case KFE_ACTIVATION_SIGMOID:
        x = sigmoid(x);
        break;
      case KFE_ACTIVATION_TANH:
        x = tanh_act(x);
        break;
      case KFE_ACTIVATION_GELU:
        x = gelu(x);
        break;
      case KFE_ACTIVATION_SWISH:
        x = swish(x);
        break;
      default:
        break;
      }
    }

    C_row[i] = x;
  }
}

/* ========================================================================== */
/* Layer Normalization with EVE                                               */
/* ========================================================================== */

static void apply_layernorm_row_eve(float *row, size_t n,
                                    const kfe_layernorm_params_t *ln_params) {
  using wide_t = eve::wide<float>;
  constexpr size_t simd_width = wide_t::size();

  // Step 1: Compute mean
  float sum = 0.0f;
  {
    wide_t sum_vec{0.0f};
    size_t i = 0;
    const size_t vec_end = (n / simd_width) * simd_width;

    for (; i < vec_end; i += simd_width) {
      wide_t x{&row[i]};
      sum_vec = sum_vec + x;
    }

    sum = eve::reduce(sum_vec);
    for (; i < n; ++i) {
      sum += row[i];
    }
  }
  float mean = sum / n;

  // Step 2: Compute variance
  float var_sum = 0.0f;
  {
    wide_t var_vec{0.0f};
    wide_t mean_vec{mean};
    size_t i = 0;
    const size_t vec_end = (n / simd_width) * simd_width;

    for (; i < vec_end; i += simd_width) {
      wide_t x{&row[i]};
      wide_t diff = x - mean_vec;
      var_vec = var_vec + (diff * diff);
    }

    var_sum = eve::reduce(var_vec);
    for (; i < n; ++i) {
      float diff = row[i] - mean;
      var_sum += diff * diff;
    }
  }
  float variance = var_sum / n;
  float inv_std = 1.0f / std::sqrt(variance + ln_params->epsilon);

  // Step 3: Normalize and scale
  {
    wide_t mean_vec{mean};
    wide_t inv_std_vec{inv_std};
    size_t i = 0;
    const size_t vec_end = (n / simd_width) * simd_width;

    for (; i < vec_end; i += simd_width) {
      wide_t x{&row[i]};
      wide_t gamma{&ln_params->gamma[i]};
      wide_t beta{&ln_params->beta[i]};

      // Normalize
      x = (x - mean_vec) * inv_std_vec;
      // Scale and shift
      x = x * gamma + beta;

      eve::store(x, &row[i]);
    }

    // Scalar tail
    for (; i < n; ++i) {
      float x = (row[i] - mean) * inv_std;
      row[i] = x * ln_params->gamma[i] + ln_params->beta[i];
    }
  }
}

} // anonymous namespace

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

extern "C" {

int kfe_sgemm_eve_epilogue(kfe_layout_t layout, kfe_transpose_t trans_a,
                           kfe_transpose_t trans_b, size_t m, size_t n,
                           size_t k, float alpha, const float *A, size_t lda,
                           const float *B, size_t ldb, const float *bias,
                           const void *norm_params, const float *residual,
                           size_t ldr, const kfe_epilogue_config_t *config,
                           float *C, size_t ldc, kfe_perf_stats_t *stats) {
  if (!g_kfe_state.initialized)
    return KFE_ERR_NOT_INITIALIZED;
  if (!A || !B || !C || !config)
    return KFE_ERR_INVALID_ARG;
  if (m == 0 || n == 0 || k == 0)
    return KFE_ERR_INVALID_ARG;

  auto t_start = std::chrono::high_resolution_clock::now();

  // Step 1: GEMM
  mil_perf_stats_t mil_stats = {};
  int result = mil_sgemm(to_mil_layout(layout), to_mil_transpose(trans_a),
                         to_mil_transpose(trans_b), m, n, k, alpha, A, lda, B,
                         ldb, 0.0f, C, ldc, &mil_stats);

  if (result != MIL_OK) {
    return KFE_ERR_INTERNAL;
  }

  // Step 2: Apply fused epilogue with EVE
  if (layout == KFE_LAYOUT_ROW_MAJOR) {
    for (size_t i = 0; i < m; ++i) {
      float *C_row = C + i * ldc;
      const float *res_row = residual ? (residual + i * ldr) : nullptr;

      // Layer normalization is applied per-row
      if (config->enable_normalization && config->norm_type == KFE_NORM_LAYER) {
        apply_layernorm_row_eve(
            C_row, n, static_cast<const kfe_layernorm_params_t *>(norm_params));
      }

      // Apply rest of epilogue
      process_row_eve_epilogue(
          C_row, n, bias,
          (config->norm_type == KFE_NORM_BATCH)
              ? static_cast<const kfe_batchnorm_params_t *>(norm_params)
              : nullptr,
          res_row, config);
    }
  } else {
    // Column-major: process by columns
    for (size_t j = 0; j < n; ++j) {
      float *C_col = C + j * ldc;
      const float *res_col = residual ? (residual + j * ldr) : nullptr;

      // TODO: Column-major implementation
      // For now, use scalar fallback
      for (size_t i = 0; i < m; ++i) {
        float x = C_col[i];

        if (config->enable_bias && bias) {
          x += bias[j];
        }

        if (config->enable_residual && res_col) {
          x += res_col[i];
        }

        if (config->enable_activation) {
          switch (config->activation) {
          case KFE_ACTIVATION_RELU:
            x = relu(x);
            break;
          default:
            break;
          }
        }

        C_col[i] = x;
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();

  // Update statistics
  g_kfe_state.total_fused_ops.fetch_add(1);
  size_t ops_count = config->enable_bias + config->enable_normalization +
                     config->enable_residual + config->enable_activation;
  size_t saved = ops_count * m * n * sizeof(float);
  g_kfe_state.total_memory_saved.fetch_add(saved);

  if (stats) {
    double elapsed =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double flops = 2.0 * m * n * k + ops_count * m * n;
    stats->gflops = (flops / elapsed) / 1e6;
    stats->elapsed_ms = elapsed;
    stats->fused_ops_count = 1 + ops_count; // GEMM + epilogue ops
    stats->memory_saved_bytes = saved;
    stats->bandwidth_gbps = mil_stats.bandwidth_gbps;
    stats->fusion_pattern = "GEMM+EVE_Epilogue";
    stats->kernel_backend = mil_stats.kernel_used;
  }

  return KFE_OK;
}

int kfe_sgemm_batchnorm_activation(kfe_layout_t layout, kfe_transpose_t trans_a,
                                   kfe_transpose_t trans_b, size_t m, size_t n,
                                   size_t k, float alpha, const float *A,
                                   size_t lda, const float *B, size_t ldb,
                                   const float *bias,
                                   const kfe_batchnorm_params_t *bn_params,
                                   kfe_activation_t activation, float *C,
                                   size_t ldc, kfe_perf_stats_t *stats) {
  kfe_epilogue_config_t config = {};
  config.enable_bias = (bias != nullptr);
  config.enable_normalization = (bn_params != nullptr);
  config.enable_activation = (activation != KFE_ACTIVATION_NONE);
  config.enable_residual = 0;
  config.norm_type = KFE_NORM_BATCH;
  config.activation = activation;
  config.use_eve_simd = 1;

  return kfe_sgemm_eve_epilogue(layout, trans_a, trans_b, m, n, k, alpha, A,
                                lda, B, ldb, bias, bn_params, nullptr, 0,
                                &config, C, ldc, stats);
}

int kfe_sgemm_layernorm_activation(kfe_layout_t layout, kfe_transpose_t trans_a,
                                   kfe_transpose_t trans_b, size_t m, size_t n,
                                   size_t k, float alpha, const float *A,
                                   size_t lda, const float *B, size_t ldb,
                                   const float *bias,
                                   const kfe_layernorm_params_t *ln_params,
                                   kfe_activation_t activation, float *C,
                                   size_t ldc, kfe_perf_stats_t *stats) {
  kfe_epilogue_config_t config = {};
  config.enable_bias = (bias != nullptr);
  config.enable_normalization = (ln_params != nullptr);
  config.enable_activation = (activation != KFE_ACTIVATION_NONE);
  config.enable_residual = 0;
  config.norm_type = KFE_NORM_LAYER;
  config.activation = activation;
  config.use_eve_simd = 1;

  return kfe_sgemm_eve_epilogue(layout, trans_a, trans_b, m, n, k, alpha, A,
                                lda, B, ldb, bias, ln_params, nullptr, 0,
                                &config, C, ldc, stats);
}

int kfe_eve_is_available(void) {
  return 1; // EVE is header-only and always available
}

size_t kfe_eve_simd_width(void) {
  using wide_t = eve::wide<float>;
  return wide_t::size();
}

const char *kfe_eve_backend_name(void) {
  static char buf[128];

  // Debug: Print what EVE actually detects
  using wide_t = eve::wide<float>;
  size_t width = wide_t::size();

#if defined(__AVX512F__)
  snprintf(buf, sizeof(buf), "EVE/AVX-512 (width=%zu)", width);
#elif defined(__AVX2__)
  snprintf(buf, sizeof(buf), "EVE/AVX2 (width=%zu)", width);
#elif defined(__AVX__)
  snprintf(buf, sizeof(buf), "EVE/AVX (width=%zu)", width);
#elif defined(__SSE4_2__)
  snprintf(buf, sizeof(buf), "EVE/SSE4.2 (width=%zu)", width);
#elif defined(__SSE2__)
  snprintf(buf, sizeof(buf), "EVE/SSE2 (width=%zu)", width);
#else
  snprintf(buf, sizeof(buf), "EVE/Scalar (width=%zu)", width);
#endif

  return buf;
}

int kfe_eve_benchmark(size_t size, int iterations) {
  // TODO: Implement comprehensive benchmarks
  printf("EVE Benchmark:\n");
  printf("  SIMD Width: %zu floats\n", kfe_eve_simd_width());
  printf("  Backend: %s\n", kfe_eve_backend_name());
  return KFE_OK;
}

} // extern "C"