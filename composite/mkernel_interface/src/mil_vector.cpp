/* composite/mkernel_interface/src/mil_vector.cpp */

#include "microkernel_interface.h"
#include "vmath_engine.h"

/* ========================================================================== */
/* Vector Operations (Single Precision - float)                               */
/* ========================================================================== */


int mil_expf(const float *x, float *y, size_t n) {
  return vmath_expf(x, y, n);
}

int mil_logf(const float *x, float *y, size_t n) {
  return vmath_logf(x, y, n);
}

int mil_log10f(const float *x, float *y, size_t n) {
  return vmath_log10f(x, y, n);
}

int mil_sinf(const float *x, float *y, size_t n) {
  return vmath_sinf(x, y, n);
}

int mil_cosf(const float *x, float *y, size_t n) {
  return vmath_cosf(x, y, n);
}

int mil_tanf(const float *x, float *y, size_t n) {
  return vmath_tanf(x, y, n);
}

int mil_asinf(const float *x, float *y, size_t n) {
  return vmath_asinf(x, y, n);
}

int mil_acosf(const float *x, float *y, size_t n) {
  return vmath_acosf(x, y, n);
}

int mil_atanf(const float *x, float *y, size_t n) {
  return vmath_atanf(x, y, n);
}

int mil_atan2f(const float *y, const float *x, float *result, size_t n) {
  return vmath_atan2f(y, x, result, n);
}

int mil_sinhf(const float *x, float *y, size_t n) {
  return vmath_sinhf(x, y, n);
}

int mil_coshf(const float *x, float *y, size_t n) {
  return vmath_coshf(x, y, n);
}

int mil_tanhf(const float *x, float *y, size_t n) {
  return vmath_tanhf(x, y, n);
}

int mil_sqrtf(const float *x, float *y, size_t n) {
  return vmath_sqrtf(x, y, n);
}

int mil_cbrtf(const float *x, float *y, size_t n) {
  return vmath_cbrtf(x, y, n);
}

int mil_powf_scalar(const float *x, float p, float *y, size_t n) {
  return vmath_powf_scalar(x, p, y, n);
}

int mil_powf(const float *x, const float *y, float *result, size_t n) {
  return vmath_powf(x, y, result, n);
}

int mil_erff(const float *x, float *y, size_t n) {
  return vmath_erff(x, y, n);
}

int mil_erfcf(const float *x, float *y, size_t n) {
  return vmath_erfcf(x, y, n);
}

int mil_gammaf(const float *x, float *y, size_t n) {
  return vmath_gammaf(x, y, n);
}

int mil_lgammaf(const float *x, float *y, size_t n) {
  return vmath_lgammaf(x, y, n);
}

/* ========================================================================== */
/* Vector Operations (Double Precision - double)                              */
/* ========================================================================== */

int mil_exp(const double *x, double *y, size_t n) {
  return vmath_exp(x, y, n);
}

int mil_log(const double *x, double *y, size_t n) {
  return vmath_log(x, y, n);
}

int mil_log10(const double *x, double *y, size_t n) {
  return vmath_log10(x, y, n);
}

int mil_sin(const double *x, double *y, size_t n) {
  return vmath_sin(x, y, n);
}

int mil_cos(const double *x, double *y, size_t n) {
  return vmath_cos(x, y, n);
}

int mil_tan(const double *x, double *y, size_t n) {
  return vmath_tan(x, y, n);
}

int mil_asin(const double *x, double *y, size_t n) {
  return vmath_asin(x, y, n);
}

int mil_acos(const double *x, double *y, size_t n) {
  return vmath_acos(x, y, n);
}

int mil_atan(const double *x, double *y, size_t n) {
  return vmath_atan(x, y, n);
}

int mil_atan2(const double *y, const double *x, double *result, size_t n) {
  return vmath_atan2(y, x, result, n);
}

int mil_sinh(const double *x, double *y, size_t n) {
  return vmath_sinh(x, y, n);
}

int mil_cosh(const double *x, double *y, size_t n) {
  return vmath_cosh(x, y, n);
}

int mil_tanh(const double *x, double *y, size_t n) {
  return vmath_tanh(x, y, n);
}

int mil_sqrt(const double *x, double *y, size_t n) {
  return vmath_sqrt(x, y, n);
}

int mil_cbrt(const double *x, double *y, size_t n) {
  return vmath_cbrt(x, y, n);
}

int mil_pow_scalar(const double *x, double p, double *y, size_t n) {
  return vmath_pow_scalar(x, p, y, n);
}

int mil_pow(const double *x, const double *y, double *result, size_t n) {
  return vmath_pow(x, y, result, n);
}

int mil_erf(const double *x, double *y, size_t n) {
  return vmath_erf(x, y, n);
}

int mil_erfc(const double *x, double *y, size_t n) {
  return vmath_erfc(x, y, n);
}

int mil_gamma(const double *x, double *y, size_t n) {
  return vmath_gamma(x, y, n);
}

int mil_lgamma(const double *x, double *y, size_t n) {
  return vmath_lgamma(x, y, n);
}

/* ========================================================================== */
/* Fused Operations (Common ML/Scientific patterns)                           */
/* ========================================================================== */

int mil_sigmoidf(const float *x, float *y, size_t n) {
  return vmath_sigmoidf(x, y, n);
}

int mil_sigmoid(const double *x, double *y, size_t n) {
  return vmath_sigmoid(x, y, n);
}

int mil_reluf(const float *x, float *y, size_t n) {
  return vmath_reluf(x, y, n);
}

int mil_relu(const double *x, double *y, size_t n) {
  return vmath_relu(x, y, n);
}

int mil_leaky_reluf(const float *x, float *y, size_t n) {
  return vmath_leaky_reluf(x, y, n);
}

int mil_leaky_relu(const double *x, double *y, size_t n) {
  return vmath_leaky_relu(x, y, n);
}

int mil_relu6f(const float *x, float *y, size_t n) {
  return vmath_relu6f(x, y, n);
}

int mil_relu6(const double *x, double *y, size_t n) {
  return vmath_relu6(x, y, n);
}

int mil_softplusf(const float *x, float *y, size_t n) {
  return vmath_softplusf(x, y, n);
}

int mil_softplus(const double *x, double *y, size_t n) {
  return vmath_softplus(x, y, n);
}

int mil_geluf(const float *x, float *y, size_t n) {
  return vmath_geluf(x, y, n);
}

int mil_gelu(const double *x, double *y, size_t n) {
  return vmath_gelu(x, y, n);
}
