#ifndef VMATH_FALLBACK_H
#define VMATH_FALLBACK_H

/**
 * @file vmath_fallback.h
 * @brief Portable scalar fallback implementations
 *
 * Pure C implementations using standard math.h functions.
 * Used when SIMD is unavailable or as reference for testing.
 *
 * These are deliberately simple and portable, not optimized.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ========================================================================== */
  /* Scalar Fallback Operations (Float)                                         */
  /* ========================================================================== */

  void fallback_expf(const float *x, float *y, size_t n);
  void fallback_logf(const float *x, float *y, size_t n);
  void fallback_log10f(const float *x, float *y, size_t n);
  void fallback_sinf(const float *x, float *y, size_t n);
  void fallback_cosf(const float *x, float *y, size_t n);
  void fallback_tanf(const float *x, float *y, size_t n);
  void fallback_asinf(const float *x, float *y, size_t n);
  void fallback_acosf(const float *x, float *y, size_t n);
  void fallback_atanf(const float *x, float *y, size_t n);
  void fallback_atan2f(const float *y, const float *x, float *result, size_t n);
  void fallback_sinhf(const float *x, float *y, size_t n);
  void fallback_coshf(const float *x, float *y, size_t n);
  void fallback_tanhf(const float *x, float *y, size_t n);
  void fallback_sqrtf(const float *x, float *y, size_t n);
  void fallback_cbrtf(const float *x, float *y, size_t n);
  void fallback_powf_scalar(const float *x, float p, float *y, size_t n);
  void fallback_powf(const float *x, const float *y, float *result, size_t n);
  void fallback_erff(const float *x, float *y, size_t n);
  void fallback_erfcf(const float *x, float *y, size_t n);
  void fallback_gammaf(const float *x, float *y, size_t n);
  void fallback_lgammaf(const float *x, float *y, size_t n);
  void fallback_log1pf(const float *x, float *y, size_t n);
  void fallback_maxf(const float *x, const float *y, float *result, size_t n);
  void fallback_minf(const float *x, const float *y, float *result, size_t n);

  /* ========================================================================== */
  /* Scalar Fallback Operations (Double)                                        */
  /* ========================================================================== */

  void fallback_exp(const double *x, double *y, size_t n);
  void fallback_log(const double *x, double *y, size_t n);
  void fallback_log10(const double *x, double *y, size_t n);
  void fallback_sin(const double *x, double *y, size_t n);
  void fallback_cos(const double *x, double *y, size_t n);
  void fallback_tan(const double *x, double *y, size_t n);
  void fallback_asin(const double *x, double *y, size_t n);
  void fallback_acos(const double *x, double *y, size_t n);
  void fallback_atan(const double *x, double *y, size_t n);
  void fallback_atan2(const double *y, const double *x, double *result, size_t n);
  void fallback_sinh(const double *x, double *y, size_t n);
  void fallback_cosh(const double *x, double *y, size_t n);
  void fallback_tanh(const double *x, double *y, size_t n);
  void fallback_sqrt(const double *x, double *y, size_t n);
  void fallback_cbrt(const double *x, double *y, size_t n);
  void fallback_pow_scalar(const double *x, double p, double *y, size_t n);
  void fallback_pow(const double *x, const double *y, double *result, size_t n);
  void fallback_erf(const double *x, double *y, size_t n);
  void fallback_erfc(const double *x, double *y, size_t n);
  void fallback_gamma(const double *x, double *y, size_t n);
  void fallback_lgamma(const double *x, double *y, size_t n);
  void fallback_log1p(const double *x, double *y, size_t n);
  void fallback_max(const double *x, const double *y, double *result, size_t n);
  void fallback_min(const double *x, const double *y, double *result, size_t n);

  /* ========================================================================== */
  /* Fused Operations Fallback                                                  */
  /* ========================================================================== */

  void fallback_sigmoidf(const float *x, float *y, size_t n);
  void fallback_sigmoid(const double *x, double *y, size_t n);
  void fallback_reluf(const float *x, float *y, size_t n);
  void fallback_relu(const double *x, double *y, size_t n);
  void fallback_softplusf(const float *x, float *y, size_t n);
  void fallback_softplus(const double *x, double *y, size_t n);
  void fallback_geluf(const float *x, float *y, size_t n);
  void fallback_gelu(const double *x, double *y, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* VMATH_FALLBACK_H */