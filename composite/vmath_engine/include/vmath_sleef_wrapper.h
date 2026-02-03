#ifndef VMATH_SLEEF_WRAPPER_H
#define VMATH_SLEEF_WRAPPER_H

/**
 * @file vmath_sleef_wrapper.h
 * @brief SLEEF library integration layer
 *
 * Wrapper around SLEEF (SIMD Library for Evaluating Elementary Functions)
 * for high-precision vectorized transcendental operations.
 *
 * SLEEF provides multiple precision modes:
 *   - ULP 1.0: Highest precision (slower)
 *   - ULP 3.5: Balanced (default)
 *   - Fast: Lower precision (faster)
 *
 * We expose the ULP 3.5 variants as default, with optional fast variants.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ========================================================================== */
  /* SLEEF Function Pointers (ISA-specific dispatch)                            */
  /* ========================================================================== */

  /**
   * Function pointer types for SLEEF kernels
   * These will be resolved at runtime based on detected ISA
   */

  /* Single precision (float) */
  typedef void (*sleef_expf_fn)(const float *, float *, size_t);
  typedef void (*sleef_logf_fn)(const float *, float *, size_t);
  typedef void (*sleef_sinf_fn)(const float *, float *, size_t);
  typedef void (*sleef_cosf_fn)(const float *, float *, size_t);
  typedef void (*sleef_powf_fn)(const float *, const float *, float *, size_t);

  /* Double precision (double) */
  typedef void (*sleef_exp_fn)(const double *, double *, size_t);
  typedef void (*sleef_log_fn)(const double *, double *, size_t);
  typedef void (*sleef_sin_fn)(const double *, double *, size_t);
  typedef void (*sleef_cos_fn)(const double *, double *, size_t);
  typedef void (*sleef_pow_fn)(const double *, const double *, double *, size_t);

  /* ========================================================================== */
  /* SLEEF Wrapper Initialization                                                */
  /* ========================================================================== */

  /**
   * @brief Initialize SLEEF wrapper with detected ISA
   * @return 0 on success, -1 on failure
   */
  int sleef_wrapper_init(void);

  /**
   * @brief Check if SLEEF is available for current ISA
   * @return 1 if available, 0 otherwise
   */
  int sleef_is_available(void);

  /* ========================================================================== */
  /* SLEEF Vector Operations (Float)                                            */
  /* ========================================================================== */

  void sleef_vec_expf(const float *x, float *y, size_t n);
  void sleef_vec_logf(const float *x, float *y, size_t n);
  void sleef_vec_log10f(const float *x, float *y, size_t n);
  void sleef_vec_sinf(const float *x, float *y, size_t n);
  void sleef_vec_cosf(const float *x, float *y, size_t n);
  void sleef_vec_tanf(const float *x, float *y, size_t n);
  void sleef_vec_asinf(const float *x, float *y, size_t n);
  void sleef_vec_acosf(const float *x, float *y, size_t n);
  void sleef_vec_atanf(const float *x, float *y, size_t n);
  void sleef_vec_atan2f(const float *y, const float *x, float *result, size_t n);
  void sleef_vec_sinhf(const float *x, float *y, size_t n);
  void sleef_vec_coshf(const float *x, float *y, size_t n);
  void sleef_vec_tanhf(const float *x, float *y, size_t n);
  void sleef_vec_sqrtf(const float *x, float *y, size_t n);
  void sleef_vec_cbrtf(const float *x, float *y, size_t n);
  void sleef_vec_powf(const float *x, const float *y, float *result, size_t n);
  void sleef_vec_erff(const float *x, float *y, size_t n);
  void sleef_vec_erfcf(const float *x, float *y, size_t n);
  void sleef_vec_maxf(const float *x, const float *y, float *result, size_t n);
  void sleef_vec_log1pf(const float *x, float *y, size_t n);
  void sleef_vec_minf(const float *x, const float *y, float *result, size_t n);

  /* ========================================================================== */
  /* SLEEF Vector Operations (Double)                                           */
  /* ========================================================================== */

  void sleef_vec_exp(const double *x, double *y, size_t n);
  void sleef_vec_log(const double *x, double *y, size_t n);
  void sleef_vec_log10(const double *x, double *y, size_t n);
  void sleef_vec_sin(const double *x, double *y, size_t n);
  void sleef_vec_cos(const double *x, double *y, size_t n);
  void sleef_vec_tan(const double *x, double *y, size_t n);
  void sleef_vec_asin(const double *x, double *y, size_t n);
  void sleef_vec_acos(const double *x, double *y, size_t n);
  void sleef_vec_atan(const double *x, double *y, size_t n);
  void sleef_vec_atan2(const double *y, const double *x, double *result, size_t n);
  void sleef_vec_sinh(const double *x, double *y, size_t n);
  void sleef_vec_cosh(const double *x, double *y, size_t n);
  void sleef_vec_tanh(const double *x, double *y, size_t n);
  void sleef_vec_sqrt(const double *x, double *y, size_t n);
  void sleef_vec_cbrt(const double *x, double *y, size_t n);
  void sleef_vec_pow(const double *x, const double *y, double *result, size_t n);
  void sleef_vec_erf(const double *x, double *y, size_t n);
  void sleef_vec_erfc(const double *x, double *y, size_t n);
  void sleef_vec_max(const double *x, const double *y, double *result, size_t n);
  void sleef_vec_min(const double *x, const double *y, double *result, size_t n);
  void sleef_vec_log1p(const double *x, double *y, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* VMATH_SLEEF_WRAPPER_H */