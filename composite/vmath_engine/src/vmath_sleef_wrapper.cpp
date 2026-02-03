// vmath_sleef_wrapper.cpp
// SLEEF library integration with ISA dispatch

#include "vmath_sleef_wrapper.h"
#include "vmath_fallback.h"
#include "cpu_features.h"
#include <cmath>
#include <cstring>

// Forward declarations of isolated processing functions from vmath_sleef_isolated.cpp
extern "C"
{
  // AVX2 Float
  size_t sleef_avx2_expf_process(const float *, float *, size_t);
  size_t sleef_avx2_logf_process(const float *, float *, size_t);
  size_t sleef_avx2_log10f_process(const float *, float *, size_t);
  size_t sleef_avx2_sinf_process(const float *, float *, size_t);
  size_t sleef_avx2_cosf_process(const float *, float *, size_t);
  size_t sleef_avx2_tanf_process(const float *, float *, size_t);
  size_t sleef_avx2_powf_process(const float *, const float *, float *, size_t);
  size_t sleef_avx2_sqrtf_process(const float *, float *, size_t);
  size_t sleef_avx2_cbrtf_process(const float *, float *, size_t);
  size_t sleef_avx2_sinhf_process(const float *, float *, size_t);
  size_t sleef_avx2_coshf_process(const float *, float *, size_t);
  size_t sleef_avx2_tanhf_process(const float *, float *, size_t);
  size_t sleef_avx2_asinf_process(const float *, float *, size_t);
  size_t sleef_avx2_acosf_process(const float *, float *, size_t);
  size_t sleef_avx2_atanf_process(const float *, float *, size_t);
  size_t sleef_avx2_atan2f_process(const float *, const float *, float *, size_t);
  size_t sleef_avx2_erff_process(const float *, float *, size_t);
  size_t sleef_avx2_erfcf_process(const float *, float *, size_t);
  size_t sleef_avx2_maxf_process(const float *x, const float *y, float *result, size_t n);
  size_t sleef_avx2_minf_process(const float *x, const float *y, float *result, size_t n);
  size_t sleef_avx2_log1pf_process(const float *x, float *y, size_t n);

  // AVX2 Double
  size_t sleef_avx2_exp_process(const double *, double *, size_t);
  size_t sleef_avx2_log_process(const double *, double *, size_t);
  size_t sleef_avx2_sin_process(const double *, double *, size_t);
  size_t sleef_avx2_asin_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_sinh_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_cos_process(const double *, double *, size_t);
  size_t sleef_avx2_acos_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_cosh_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_tan_process(const double *, double *, size_t);
  size_t sleef_avx2_atan_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_atan2_process(const double *y, const double *x, double *result, size_t n);
  size_t sleef_avx2_tanh_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_pow_process(const double *, const double *, double *, size_t);
  size_t sleef_avx2_sqrt_process(const double *, double *, size_t);
  size_t sleef_avx2_log10_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_cbrt_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_erf_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_erfc_process(const double *x, double *y, size_t n);
  size_t sleef_avx2_max_process(const double *x, const double *y, double *result, size_t n);
  size_t sleef_avx2_min_process(const double *x, const double *y, double *result, size_t n);
  size_t sleef_avx2_log1p_process(const double *x, double *y, size_t n);
}

// Global state
static bool g_sleef_initialized = false;
static bool g_has_avx2 = false;
static bool g_has_avx512 = false;

// ============================================================================
// Initialization
// ============================================================================

int sleef_wrapper_init(void)
{
  if (g_sleef_initialized)
  {
    return 0; // Already initialized
  }

  // Detect CPU features
  CPUFeatures features = detect_cpu_features();
  g_has_avx2 = features.avx2;
  g_has_avx512 = features.avx512;

  g_sleef_initialized = true;
  return 0;
}

int sleef_is_available(void)
{
  return g_sleef_initialized && (g_has_avx2 || g_has_avx512);
}

// ============================================================================
// Float Operations - AVX2 with scalar tail
// ============================================================================

void sleef_vec_expf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_expf(x, y, n);
    return;
  }

  // Process vectorized chunks
  size_t processed = sleef_avx2_expf_process(x, y, n);

  // Scalar tail
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::exp(x[i]);
  }
}

void sleef_vec_logf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_logf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_logf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::log(x[i]);
  }
}

void sleef_vec_log10f(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_log10f(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_log10f_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::log10(x[i]);
  }
}

void sleef_vec_sinf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_sinf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_sinf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::sin(x[i]);
  }
}

void sleef_vec_cosf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_cosf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_cosf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::cos(x[i]);
  }
}

void sleef_vec_tanf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_tanf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_tanf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::tan(x[i]);
  }
}

void sleef_vec_asinf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_asinf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_asinf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::asin(x[i]);
  }
}

void sleef_vec_acosf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_acosf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_acosf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::acos(x[i]);
  }
}

void sleef_vec_atanf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_atanf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_atanf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::atan(x[i]);
  }
}

void sleef_vec_atan2f(const float *y_in, const float *x, float *result, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_atan2f(y_in, x, result, n);
    return;
  }

  size_t processed = sleef_avx2_atan2f_process(y_in, x, result, n);
  for (size_t i = processed; i < n; ++i)
  {
    result[i] = std::atan2(y_in[i], x[i]);
  }
}

void sleef_vec_sinhf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_sinhf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_sinhf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::sinh(x[i]);
  }
}

void sleef_vec_coshf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_coshf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_coshf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::cosh(x[i]);
  }
}

void sleef_vec_tanhf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_tanhf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_tanhf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::tanh(x[i]);
  }
}

void sleef_vec_sqrtf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_sqrtf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_sqrtf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::sqrt(x[i]);
  }
}

void sleef_vec_cbrtf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_cbrtf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_cbrtf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::cbrtf(x[i]);
  }
}

void sleef_vec_powf(const float *x, const float *y_in, float *result, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_powf(x, y_in, result, n);
    return;
  }

  size_t processed = sleef_avx2_powf_process(x, y_in, result, n);
  for (size_t i = processed; i < n; ++i)
  {
    result[i] = std::pow(x[i], y_in[i]);
  }
}

void sleef_vec_erff(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_erff(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_erff_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::erff(x[i]);
  }
}

void sleef_vec_erfcf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_erfcf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_erfcf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::erfcf(x[i]);
  }
}

void sleef_vec_log1pf(const float *x, float *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_log1pf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_log1pf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::log1pf(x[i]);
  }
}

void sleef_vec_maxf(const float *x, const float *y, float *result, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2) {
    fallback_maxf(x, y, result, n);
    return;
  }

  size_t processed = sleef_avx2_maxf_process(x, y, result, n);
  for (size_t i = processed; i < n; ++i) {
    result[i] = std::max(x[i], y[i]);
  }
}

void sleef_vec_minf(const float *x, const float *y, float *result, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2) {
    fallback_minf(x, y, result, n);
    return;
  }

  size_t processed = sleef_avx2_minf_process(x, y, result, n);
  for (size_t i = processed; i < n; ++i) {
    result[i] = std::min(x[i], y[i]);
  }
}

// ============================================================================
// Double Operations - AVX2 with scalar tail
// ============================================================================

void sleef_vec_max(const double *x, const double *y, double *result, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2) {
    fallback_max(x, y, result, n);
    return;
  }

  size_t processed = sleef_avx2_max_process(x, y, result, n);
  for (size_t i = processed; i < n; ++i) {
    result[i] = std::max(x[i], y[i]);
  }
}

void sleef_vec_min(const double *x, const double *y, double *result, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2) {
    fallback_min(x, y, result, n);
    return;
  }

  size_t processed = sleef_avx2_min_process(x, y, result, n);
  for (size_t i = processed; i < n; ++i) {
    result[i] = std::min(x[i], y[i]);
  }
}

void sleef_vec_log1p(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_log1p(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_log1p_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::log1p(x[i]);
  }
}

void sleef_vec_exp(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_exp(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_exp_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::exp(x[i]);
  }
}

void sleef_vec_log(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_log(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_log_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::log(x[i]);
  }
}

void sleef_vec_log10(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_log10(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_log10_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::log(x[i]);
  }
}

void sleef_vec_sin(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_sin(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_sin_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::sin(x[i]);
  }
}

void sleef_vec_cos(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_cos(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_cos_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::cos(x[i]);
  }
}

void sleef_vec_tan(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_tan(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_tan_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::tan(x[i]);
  }
}

void sleef_vec_asin(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_asin(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_asin_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::asin(x[i]);
  }
}

void sleef_vec_acos(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_acos(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_acos_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::acos(x[i]);
  }
}

void sleef_vec_atan(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_atan(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_atan_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::atan(x[i]);
  }
}

void sleef_vec_atan2(const double *y_in, const double *x, double *result, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_atan2(y_in, x, result, n);
    return;
  }

  size_t processed = sleef_avx2_atan2_process(y_in, x, result, n);
  for (size_t i = processed; i < n; ++i)
  {
    result[i] = std::atan2(x[i], y_in[i]);
  }
}

void sleef_vec_sinh(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_sinh(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_sinh_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::sinh(x[i]);
  }
}

void sleef_vec_cosh(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_cosh(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_cosh_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::cosh(x[i]);
  }
}

void sleef_vec_tanh(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_tanh(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_tanh_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::tanh(x[i]);
  }
}

void sleef_vec_sqrt(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_sqrt(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_sqrt_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::sqrt(x[i]);
  }
}

void sleef_vec_cbrt(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_cbrt(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_cbrt_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::cbrt(x[i]);
  }
}

void sleef_vec_pow(const double *x, const double *y_in, double *result, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_pow(x, y_in, result, n);
    return;
  }

  size_t processed = sleef_avx2_pow_process(x, y_in, result, n);
  for (size_t i = processed; i < n; ++i)
  {
    result[i] = std::pow(x[i], y_in[i]);
  }
}

void sleef_vec_erf(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_erf(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_erf_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::erf(x[i]);
  }
}

void sleef_vec_erfc(const double *x, double *y, size_t n)
{
  if (!g_sleef_initialized || !g_has_avx2)
  {
    fallback_erfc(x, y, n);
    return;
  }

  size_t processed = sleef_avx2_erfc_process(x, y, n);
  for (size_t i = processed; i < n; ++i)
  {
    y[i] = std::erfc(x[i]);
  }
}