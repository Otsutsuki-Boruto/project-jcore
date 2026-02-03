// advanced/kFusion_engine/src/kernel_fusion_helpers.cpp

#include "kernel_fusion_engine_internal.h"
#include "jcore_isa_dispatch.h"
#include "microkernel_interface.h"

namespace kfe_internal
{

  /* ========================================================================== */
  /* Vectorized Activation Implementations (AVX2)                               */
  /* ========================================================================== */

#ifdef __AVX2__

  __m256 avx2_relu(__m256 x)
  {
    __m256 zero = _mm256_setzero_ps();
    return _mm256_max_ps(x, zero);
  }

  __m256 avx2_relu6(__m256 x)
  {
    __m256 zero = _mm256_setzero_ps();
    __m256 six = _mm256_set1_ps(6.0f);
    return _mm256_min_ps(_mm256_max_ps(x, zero), six);
  }

  __m256 avx2_leaky_relu(__m256 x)
  {
    __m256 zero = _mm256_setzero_ps();
    __m256 alpha = _mm256_set1_ps(0.01f);
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    __m256 neg_part = _mm256_mul_ps(x, alpha);
    return _mm256_blendv_ps(neg_part, x, mask);
  }

  __m256 avx2_sigmoid(__m256 x)
  {
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 x_half = _mm256_mul_ps(x, half);

    // Clamp to prevent overflow
    __m256 clamp_val = _mm256_set1_ps(5.0f);
    __m256 neg_clamp = _mm256_set1_ps(-5.0f);
    x_half = _mm256_min_ps(_mm256_max_ps(x_half, neg_clamp), clamp_val);

    // tanh approximation: x * (27 + x^2) / (27 + 9*x^2)
    __m256 x2 = _mm256_mul_ps(x_half, x_half);
    __m256 c27 = _mm256_set1_ps(27.0f);
    __m256 c9 = _mm256_set1_ps(9.0f);

    __m256 num = _mm256_mul_ps(x_half, _mm256_add_ps(c27, x2));
    __m256 den = _mm256_add_ps(c27, _mm256_mul_ps(c9, x2));
    __m256 tanh_val = _mm256_div_ps(num, den);

    return _mm256_add_ps(half, _mm256_mul_ps(half, tanh_val));
  }

#endif // __AVX2__

  /* ========================================================================== */
  /* Apply Activation - Scalar Version                                          */
  /* ========================================================================== */

  void apply_activation_scalar(float *data, size_t n, kfe_activation_t act)
  {
    switch (act)
    {
    case KFE_ACTIVATION_RELU:
      for (size_t i = 0; i < n; ++i)
        data[i] = relu(data[i]);
      break;
    case KFE_ACTIVATION_RELU6:
      for (size_t i = 0; i < n; ++i)
        data[i] = relu6(data[i]);
      break;
    case KFE_ACTIVATION_TANH:
      for (size_t i = 0; i < n; ++i)
        data[i] = tanh_act(data[i]);
      break;
    case KFE_ACTIVATION_SIGMOID:
      for (size_t i = 0; i < n; ++i)
        data[i] = sigmoid(data[i]);
      break;
    case KFE_ACTIVATION_GELU:
      for (size_t i = 0; i < n; ++i)
        data[i] = gelu(data[i]);
      break;
    case KFE_ACTIVATION_SWISH:
      for (size_t i = 0; i < n; ++i)
        data[i] = swish(data[i]);
      break;
    case KFE_ACTIVATION_LEAKY_RELU:
      for (size_t i = 0; i < n; ++i)
        data[i] = leaky_relu(data[i]);
      break;
    default:
      break; // KFE_ACTIVATION_NONE
    }
  }

  /* ========================================================================== */
  /* Apply Activation - Vectorized Version                                      */
  /* ========================================================================== */

  void apply_activation_vectorized(float *data, size_t n, kfe_activation_t act)
  {
#ifdef __AVX2__
    jcore_features_t features = jcore_get_host_features();
    if (!(features & JCORE_FEAT_AVX2))
    {
      apply_activation_scalar(data, n, act);
      return;
    }

    size_t vec_n = (n / 8) * 8;
    size_t i = 0;

    switch (act)
    {
    case KFE_ACTIVATION_RELU:
      for (; i < vec_n; i += 8)
      {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = avx2_relu(v);
        _mm256_storeu_ps(&data[i], v);
      }
      break;
    case KFE_ACTIVATION_RELU6:
      for (; i < vec_n; i += 8)
      {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = avx2_relu6(v);
        _mm256_storeu_ps(&data[i], v);
      }
      break;
    case KFE_ACTIVATION_LEAKY_RELU:
      for (; i < vec_n; i += 8)
      {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = avx2_leaky_relu(v);
        _mm256_storeu_ps(&data[i], v);
      }
      break;
    case KFE_ACTIVATION_SIGMOID:
      for (; i < vec_n; i += 8)
      {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = avx2_sigmoid(v);
        _mm256_storeu_ps(&data[i], v);
      }
      break;
    case KFE_ACTIVATION_TANH:
        mil_tanhf(data, data, n);
        break;

    case KFE_ACTIVATION_GELU:
        mil_geluf(data, data, n);
        break;
    default:
      // For complex activations, use scalar
      apply_activation_scalar(data, n, act);
      return;
    }

    // Tail processing
    for (; i < n; ++i)
    {
      switch (act)
      {
      case KFE_ACTIVATION_RELU:
        data[i] = relu(data[i]);
        break;
      case KFE_ACTIVATION_RELU6:
        data[i] = relu6(data[i]);
        break;
      case KFE_ACTIVATION_LEAKY_RELU:
        data[i] = leaky_relu(data[i]);
        break;
      case KFE_ACTIVATION_SIGMOID:
        data[i] = sigmoid(data[i]);
        break;
      default:
        break;
      }
    }
#else
    apply_activation_scalar(data, n, act);
#endif
  }

  /* ========================================================================== */
  /* Bias Addition                                                               */
  /* ========================================================================== */

  void add_bias_column_major(float *C, size_t m, size_t n, size_t ldc, const float *bias)
  {
#ifdef __AVX2__
    jcore_features_t features = jcore_get_host_features();
    if (features & JCORE_FEAT_AVX2)
    {
      for (size_t j = 0; j < n; ++j)
      {
        float b = bias[j];
        size_t i = 0;
        size_t vec_m = (m / 8) * 8; // process in blocks of 8 floats
        for (; i < vec_m; i += 8)
        {
          __m256 col_vec = _mm256_loadu_ps(&C[i + j * ldc]);
          __m256 bias_vec = _mm256_set1_ps(b);
          col_vec = _mm256_add_ps(col_vec, bias_vec);
          _mm256_storeu_ps(&C[i + j * ldc], col_vec);
        }
        // Tail loop for remaining elements
        for (; i < m; ++i)
        {
          C[i + j * ldc] += b;
        }
      }
      return;
    }
#endif
    // Scalar fallback for non-AVX2 CPUs
    for (size_t j = 0; j < n; ++j)
    {
      float b = bias[j];
      for (size_t i = 0; i < m; ++i)
      {
        C[i + j * ldc] += b;
      }
    }
  }


  void add_bias_row_major(float *C, size_t m, size_t n, size_t ldc, const float *bias)
  {
#ifdef __AVX2__
    jcore_features_t features = jcore_get_host_features();
    if (features & JCORE_FEAT_AVX2)
    {
      for (size_t i = 0; i < m; ++i)
      {
        float *row = C + i * ldc;
        size_t j = 0;
        size_t vec_n = (n / 8) * 8;  // process in blocks of 8 floats
        for (; j < vec_n; j += 8)
        {
          __m256 row_vec = _mm256_loadu_ps(&row[j]);
          __m256 bias_vec = _mm256_loadu_ps(&bias[j]);
          row_vec = _mm256_add_ps(row_vec, bias_vec);
          _mm256_storeu_ps(&row[j], row_vec);
        }
        // Tail loop for remaining elements
        for (; j < n; ++j)
        {
          row[j] += bias[j];
        }
      }
      return;
    }
#endif
    // Scalar fallback for non-AVX2 CPUs
    for (size_t i = 0; i < m; ++i)
    {
      float *row = C + i * ldc;
      for (size_t j = 0; j < n; ++j)
      {
        row[j] += bias[j];
      }
    }
  }

  /* ========================================================================== */
  /* Element-wise Operations                                                     */
  /* ========================================================================== */

  void elementwise_add(float *C, size_t m, size_t n, size_t ldc,
                       const float *D, size_t ldd, float beta)
  {
#ifdef __AVX2__
    jcore_features_t features = jcore_get_host_features();
    if (features & JCORE_FEAT_AVX2)
    {
      __m256 beta_vec = _mm256_set1_ps(beta);
      for (size_t i = 0; i < m; ++i)
      {
        float *c_row = C + i * ldc;
        const float *d_row = D + i * ldd;
        size_t j = 0;
        size_t vec_n = (n / 8) * 8;
        for (; j < vec_n; j += 8)
        {
          __m256 c_vec = _mm256_loadu_ps(&c_row[j]);
          __m256 d_vec = _mm256_loadu_ps(&d_row[j]);
          c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(beta_vec, d_vec));
          _mm256_storeu_ps(&c_row[j], c_vec);
        }
        // Tail
        for (; j < n; ++j)
        {
          c_row[j] += beta * d_row[j];
        }
      }
      return;
    }
#endif
    // Scalar fallback
    for (size_t i = 0; i < m; ++i)
    {
      float *c_row = C + i * ldc;
      const float *d_row = D + i * ldd;
      for (size_t j = 0; j < n; ++j)
      {
        c_row[j] += beta * d_row[j];
      }
    }
  }

  void elementwise_mul(float *C, size_t m, size_t n, size_t ldc,
                     const float *D, size_t ldd, float beta)
  {
#ifdef __AVX2__
    jcore_features_t features = jcore_get_host_features();
    if (features & JCORE_FEAT_AVX2)
    {
      __m256 beta_vec = _mm256_set1_ps(beta);
      for (size_t i = 0; i < m; ++i)
      {
        float *c_row = C + i * ldc;
        const float *d_row = D + i * ldd;
        size_t j = 0;
        size_t vec_n = (n / 8) * 8;

        for (; j < vec_n; j += 8)
        {
          __m256 c_vec = _mm256_loadu_ps(&c_row[j]);
          __m256 d_vec = _mm256_loadu_ps(&d_row[j]);
          c_vec = _mm256_mul_ps(c_vec, _mm256_mul_ps(beta_vec, d_vec));
          _mm256_storeu_ps(&c_row[j], c_vec);
        }

        // Tail loop
        for (; j < n; ++j)
        {
          c_row[j] *= beta * d_row[j];
        }
      }
      return;
    }
#endif

    // Scalar fallback
    for (size_t i = 0; i < m; ++i)
    {
      float *c_row = C + i * ldc;
      const float *d_row = D + i * ldd;
      for (size_t j = 0; j < n; ++j)
      {
        c_row[j] *= beta * d_row[j];
      }
    }
  }


} // namespace kfe_internal