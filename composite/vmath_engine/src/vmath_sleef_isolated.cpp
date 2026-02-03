// vmath_sleef_isolated.cpp
//
// ISA-isolated vector processing wrappers for SLEEF operations
// This file uses __attribute__((target(...))) to confine ISA-specific opcodes
// Each function processes only full vector chunks and returns elements
// processed

#include "vmath_engine.h"
#include "vmath_fallback.h"
#include "vmath_sleef_wrapper.h"
#include <cstddef>
#include <immintrin.h>
#include <sleef.h>

// Compiler feature detection
#if defined(__GNUC__) || defined(__clang__)
#define HAS_TARGET_ATTRIBUTE 1
#define TARGET_AVX2 __attribute__((target("avx2,fma")))
#define TARGET_AVX512 __attribute__((target("avx512f")))
#define NOINLINE __attribute__((noinline))
#else
#define HAS_TARGET_ATTRIBUTE 0
#define TARGET_AVX2
#define TARGET_AVX512
#define NOINLINE
#endif

extern "C" {

// ========================================================================
// AVX2 Isolated Processing (Float)
// ========================================================================

TARGET_AVX2 NOINLINE size_t sleef_avx2_expf_process(const float *x, float *y,
                                                    size_t n) {
  const size_t vec_size = 8; // AVX2: 256-bit / 32-bit = 8 floats
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_expf8_u10avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_logf_process(const float *x, float *y,
                                                    size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_logf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_log10f_process(const float *x, float *y,
                                                      size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_log10f8_u10avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_sinf_process(const float *x, float *y,
                                                    size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_sinf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_cosf_process(const float *x, float *y,
                                                    size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_cosf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_tanf_process(const float *x, float *y,
                                                    size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_tanf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_powf_process(const float *x,
                                                    const float *p, float *y,
                                                    size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vp = _mm256_loadu_ps(p + i);
    __m256 vy = Sleef_powf8_u10avx2(vx, vp);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_sqrtf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_sqrtf8_u05avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_cbrtf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_cbrtf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_sinhf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_sinhf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_coshf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_coshf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_tanhf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_tanhf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_asinf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_asinf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_acosf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_acosf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_atanf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_atanf8_u35avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_atan2f_process(const float *y_in,
                                                      const float *x,
                                                      float *result, size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vy = _mm256_loadu_ps(y_in + i);
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vr = Sleef_atan2f8_u35avx2(vy, vx);
    _mm256_storeu_ps(result + i, vr);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_erff_process(const float *x, float *y,
                                                    size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_erff8_u10avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_erfcf_process(const float *x, float *y,
                                                     size_t n) {
  const size_t vec_size = 8;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_erfcf8_u15avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_log1pf_process(const float *x, float *y, size_t n) {
  const size_t vec_size = 8; // AVX2: 256-bit / 32-bit float = 8 floats
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = Sleef_log1pf8_u10avx2(vx);
    _mm256_storeu_ps(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_maxf_process(const float *x, const float *y, float *result, size_t n) {
  const size_t vec_size = 8; // AVX2: 256-bit / 32-bit float = 8 floats
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    __m256 vr = Sleef_fmaxf8_avx2(vx, vy);
    _mm256_storeu_ps(result + i, vr);
    processed += vec_size;
  }

  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_minf_process(const float *x, const float *y, float *result, size_t n) {
  const size_t vec_size = 8; // AVX2: 256-bit / 32-bit float = 8 floats
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    __m256 vr = Sleef_fminf8_avx2(vx, vy);
    _mm256_storeu_ps(result + i, vr);
    processed += vec_size;
  }

  return processed;
}


// ========================================================================
// AVX2 Isolated Processing (Double)
// ========================================================================

TARGET_AVX2 NOINLINE size_t sleef_avx2_exp_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4; // AVX2: 256-bit / 64-bit = 4 doubles
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_expd4_u10avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_log_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_logd4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_log10_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_log10d4_u10avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_sin_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_sind4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_asin_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_asind4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_sinh_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_sinhd4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_cos_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_cosd4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_acos_process(const double *x, double *y,
                                                     size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_acosd4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_cosh_process(const double *x, double *y,
                                                     size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_coshd4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_tan_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_tand4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_atan_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_atand4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_atan2_process(const double *y, const double *x, double *result, size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vy = _mm256_loadu_pd(y + i);
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vr = Sleef_atan2d4_u35avx2(vy, vx);
    _mm256_storeu_pd(result + i, vr);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_tanh_process(const double *x, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_tanhd4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_cbrt_process(const double *x, double *y, size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_cbrtd4_u35avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_erf_process(const double *x, double *y, size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_erfd4_u10avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_erfc_process(const double *x, double *y, size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_erfcd4_u15avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_pow_process(const double *x,
                                                   const double *p, double *y,
                                                   size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vp = _mm256_loadu_pd(p + i);
    __m256d vy = Sleef_powd4_u10avx2(vx, vp);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_sqrt_process(const double *x, double *y,
                                                    size_t n) {
  const size_t vec_size = 4;
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_sqrtd4_u05avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_log1p_process(const double *x, double *y, size_t n) {
  const size_t vec_size = 4; // AVX2: 256-bit / 64-bit double = 4 doubles
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = Sleef_log1pd4_u10avx2(vx);
    _mm256_storeu_pd(y + i, vy);
    processed += vec_size;
  }
  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_max_process(const double *x, const double *y, double *result, size_t n) {
  const size_t vec_size = 4; // AVX2: 256-bit / 64-bit double = 4 doubles
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = _mm256_loadu_pd(y + i);
    __m256d vr = Sleef_fmaxd4_avx2(vx, vy);
    _mm256_storeu_pd(result + i, vr);
    processed += vec_size;
  }

  return processed;
}

TARGET_AVX2 NOINLINE size_t sleef_avx2_min_process(const double *x, const double *y, double *result, size_t n) {
  const size_t vec_size = 4; // AVX2: 256-bit / 64-bit double = 4 doubles
  size_t processed = 0;

  for (size_t i = 0; i + vec_size <= n; i += vec_size) {
    __m256d vx = _mm256_loadu_pd(x + i);
    __m256d vy = _mm256_loadu_pd(y + i);
    __m256d vr = Sleef_fmind4_avx2(vx, vy);
    _mm256_storeu_pd(result + i, vr);
    processed += vec_size;
  }

  return processed;
}

} // extern "C"