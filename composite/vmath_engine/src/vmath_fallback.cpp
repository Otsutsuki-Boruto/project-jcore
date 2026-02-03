// vmath_fallback.cpp
// Portable scalar fallback implementations using standard math.h

#include "vmath_fallback.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Single Precision (Float) Fallbacks
// ============================================================================

void fallback_expf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::exp(x[i]);
  }
}

void fallback_logf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::log(x[i]);
  }
}

void fallback_log10f(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::log10(x[i]);
  }
}

void fallback_sinf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::sin(x[i]);
  }
}

void fallback_cosf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::cos(x[i]);
  }
}

void fallback_tanf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::tan(x[i]);
  }
}

void fallback_asinf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::asin(x[i]);
  }
}

void fallback_acosf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::acos(x[i]);
  }
}

void fallback_atanf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::atan(x[i]);
  }
}

void fallback_atan2f(const float *y_in, const float *x, float *result, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    result[i] = std::atan2(y_in[i], x[i]);
  }
}

void fallback_sinhf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::sinh(x[i]);
  }
}

void fallback_coshf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::cosh(x[i]);
  }
}

void fallback_tanhf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::tanh(x[i]);
  }
}

void fallback_sqrtf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::sqrt(x[i]);
  }
}

void fallback_cbrtf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::cbrtf(x[i]);
  }
}

void fallback_powf_scalar(const float *x, float p, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::pow(x[i], p);
  }
}

void fallback_powf(const float *x, const float *y_in, float *result, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    result[i] = std::pow(x[i], y_in[i]);
  }
}

void fallback_erff(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::erff(x[i]);
  }
}

void fallback_erfcf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::erfcf(x[i]);
  }
}

void fallback_gammaf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::tgammaf(x[i]);
  }
}

void fallback_lgammaf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::lgammaf(x[i]);
  }
}

void fallback_log1pf(const float *x, float *y, size_t n) {
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::log1pf(x[i]);
  }
}


void fallback_maxf(const float *x, const float *y, float *result, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    result[i] = std::max(x[i], y[i]);
  }
}

void fallback_minf(const float *x, const float *y, float *result, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    result[i] = std::min(x[i], y[i]);
  }
}

// ============================================================================
// Double Precision Fallbacks
// ============================================================================

void fallback_log1p(const double *x, double *y, size_t n) {
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::log1p(x[i]);
  }
}


void fallback_max(const double *x, const double *y, double *result, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    result[i] = std::max(x[i], y[i]);
  }
}

void fallback_min(const double *x, const double *y, double *result, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    result[i] = std::min(x[i], y[i]);
  }
}

void fallback_exp(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::exp(x[i]);
  }
}

void fallback_log(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::log(x[i]);
  }
}

void fallback_log10(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::log10(x[i]);
  }
}

void fallback_sin(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::sin(x[i]);
  }
}

void fallback_cos(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::cos(x[i]);
  }
}

void fallback_tan(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::tan(x[i]);
  }
}

void fallback_asin(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::asin(x[i]);
  }
}

void fallback_acos(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::acos(x[i]);
  }
}

void fallback_atan(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::atan(x[i]);
  }
}

void fallback_atan2(const double *y_in, const double *x, double *result, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    result[i] = std::atan2(y_in[i], x[i]);
  }
}

void fallback_sinh(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::sinh(x[i]);
  }
}

void fallback_cosh(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::cosh(x[i]);
  }
}

void fallback_tanh(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::tanh(x[i]);
  }
}

void fallback_sqrt(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::sqrt(x[i]);
  }
}

void fallback_cbrt(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::cbrt(x[i]);
  }
}

void fallback_pow_scalar(const double *x, double p, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::pow(x[i], p);
  }
}

void fallback_pow(const double *x, const double *y_in, double *result, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    result[i] = std::pow(x[i], y_in[i]);
  }
}

void fallback_erf(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::erf(x[i]);
  }
}

void fallback_erfc(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::erfc(x[i]);
  }
}

void fallback_gamma(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::tgamma(x[i]);
  }
}

void fallback_lgamma(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::lgamma(x[i]);
  }
}

// ============================================================================
// Fused Operations Fallback
// ============================================================================

void fallback_sigmoidf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = 1.0f / (1.0f + std::exp(-x[i]));
  }
}

void fallback_sigmoid(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = 1.0 / (1.0 + std::exp(-x[i]));
  }
}

void fallback_reluf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::max(0.0f, x[i]);
  }
}

void fallback_relu(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::max(0.0, x[i]);
  }
}

void fallback_softplusf(const float *x, float *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::log1pf(std::exp(x[i]));
  }
}

void fallback_softplus(const double *x, double *y, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    y[i] = std::log1p(std::exp(x[i]));
  }
}

void fallback_geluf(const float *x, float *y, size_t n)
{
  // GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
  const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
  const float coeff = 0.044715f;

  for (size_t i = 0; i < n; ++i)
  {
    float xi = x[i];
    float xi3 = xi * xi * xi;
    float inner = sqrt_2_over_pi * (xi + coeff * xi3);
    y[i] = 0.5f * xi * (1.0f + std::tanh(inner));
  }
}

void fallback_gelu(const double *x, double *y, size_t n)
{
  const double sqrt_2_over_pi = 0.7978845608028654;
  const double coeff = 0.044715;

  for (size_t i = 0; i < n; ++i)
  {
    double xi = x[i];
    double xi3 = xi * xi * xi;
    double inner = sqrt_2_over_pi * (xi + coeff * xi3);
    y[i] = 0.5 * xi * (1.0 + std::tanh(inner));
  }
}