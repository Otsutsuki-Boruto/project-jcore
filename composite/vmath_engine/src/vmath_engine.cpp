// vmath_engine.cpp
// Main Vector Math Engine implementation with ISA dispatch

#include "vmath_engine.h"
#include "vmath_sleef_wrapper.h"
#include "vmath_fallback.h"
#include "cpu_features.h"
#include "jcore_isa_dispatch.h"
#include <cstdio>
#include <cmath>

// Global state
static bool g_vmath_initialized = false;
static vmath_isa_level_t g_current_isa = VMATH_ISA_SCALAR;

// ============================================================================
// Initialization & Query
// ============================================================================

int vmath_init(void)
{
  if (g_vmath_initialized)
  {
    return VMATH_OK; // Already initialized
  }

  // Initialize dependencies
  int ret = jcore_init_dispatch();
  if (ret != JCORE_OK)
  {
    fprintf(stderr, "[VMATH] Failed to initialize ISA dispatch\n");
    return VMATH_ERR_INTERNAL;
  }

  // Detect CPU features
  CPUFeatures features = detect_cpu_features();

  // Determine the best ISA level
  if (features.avx512)
  {
    g_current_isa = VMATH_ISA_AVX512;
  }
  else if (features.avx2)
  {
    g_current_isa = VMATH_ISA_AVX2;
  }
  else if (features.avx)
  {
    g_current_isa = VMATH_ISA_AVX;
  }
  else
  {
    g_current_isa = VMATH_ISA_SCALAR;
  }

  // Initialize SLEEF wrapper
  ret = sleef_wrapper_init();
  if (ret != 0)
  {
    fprintf(stderr, "[VMATH] SLEEF initialization failed, using fallback\n");
    g_current_isa = VMATH_ISA_SCALAR;
  }

  g_vmath_initialized = true;

  printf("[VMATH] Initialized with ISA level: %s\n", vmath_isa_name(g_current_isa));
  return VMATH_OK;
}

void vmath_shutdown(void)
{
  g_vmath_initialized = false;
  g_current_isa = VMATH_ISA_SCALAR;
}

vmath_isa_level_t vmath_get_isa_level(void)
{
  return g_current_isa;
}

const char *vmath_isa_name(vmath_isa_level_t level)
{
  switch (level)
  {
  case VMATH_ISA_SCALAR:
    return "Scalar";
  case VMATH_ISA_SSE2:
    return "SSE2";
  case VMATH_ISA_AVX:
    return "AVX";
  case VMATH_ISA_AVX2:
    return "AVX2";
  case VMATH_ISA_AVX512:
    return "AVX-512";
  default:
    return "Unknown";
  }
}

// ============================================================================
// Helper: Validate inputs
// ============================================================================

static inline int validate_inputs(const void *x, const void *y, size_t n)
{
  if (!g_vmath_initialized)
  {
    return VMATH_ERR_NOT_INITIALIZED;
  }
  if (!x || !y || n == 0)
  {
    return VMATH_ERR_INVALID_ARG;
  }
  return VMATH_OK;
}

// ============================================================================
// Single Precision (Float) Operations
// ============================================================================

int vmath_expf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_expf(x, y, n);
  }
  else
  {
    fallback_expf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_logf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_logf(x, y, n);
  }
  else
  {
    fallback_logf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_log10f(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_log10f(x, y, n);
  }
  else
  {
    fallback_log10f(x, y, n);
  }
  return VMATH_OK;
}

int vmath_sinf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_sinf(x, y, n);
  }
  else
  {
    fallback_sinf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_cosf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_cosf(x, y, n);
  }
  else
  {
    fallback_cosf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_tanf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_tanf(x, y, n);
  }
  else
  {
    fallback_tanf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_asinf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_asinf(x, y, n);
  }
  else
  {
    fallback_asinf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_acosf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_acosf(x, y, n);
  }
  else
  {
    fallback_acosf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_atanf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_atanf(x, y, n);
  }
  else
  {
    fallback_atanf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_atan2f(const float *y_in, const float *x, float *result, size_t n)
{
  if (!g_vmath_initialized)
    return VMATH_ERR_NOT_INITIALIZED;
  if (!y_in || !x || !result || n == 0)
    return VMATH_ERR_INVALID_ARG;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_atan2f(y_in, x, result, n);
  }
  else
  {
    fallback_atan2f(y_in, x, result, n);
  }
  return VMATH_OK;
}

int vmath_sinhf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_sinhf(x, y, n);
  }
  else
  {
    fallback_sinhf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_coshf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_coshf(x, y, n);
  }
  else
  {
    fallback_coshf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_tanhf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_tanhf(x, y, n);
  }
  else
  {
    fallback_tanhf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_sqrtf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_sqrtf(x, y, n);
  }
  else
  {
    fallback_sqrtf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_cbrtf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_cbrtf(x, y, n);
  }
  else
  {
    fallback_cbrtf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_powf_scalar(const float *x, float p, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  fallback_powf_scalar(x, p, y, n);
  return VMATH_OK;
}

int vmath_powf(const float *x, const float *y_in, float *result, size_t n)
{
  if (!g_vmath_initialized)
    return VMATH_ERR_NOT_INITIALIZED;
  if (!x || !y_in || !result || n == 0)
    return VMATH_ERR_INVALID_ARG;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_powf(x, y_in, result, n);
  }
  else
  {
    fallback_powf(x, y_in, result, n);
  }
  return VMATH_OK;
}

int vmath_erff(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_erff(x, y, n);
  }
  else
  {
    fallback_erff(x, y, n);
  }
  return VMATH_OK;
}

int vmath_erfcf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_erfcf(x, y, n);
  }
  else
  {
    fallback_erfcf(x, y, n);
  }
  return VMATH_OK;
}

int vmath_gammaf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  fallback_gammaf(x, y, n);
  return VMATH_OK;
}

int vmath_lgammaf(const float *x, float *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  fallback_lgammaf(x, y, n);
  return VMATH_OK;
}

// ============================================================================
// Double Precision Operations
// ============================================================================

int vmath_exp(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_exp(x, y, n);
  }
  else
  {
    fallback_exp(x, y, n);
  }
  return VMATH_OK;
}

int vmath_log(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_log(x, y, n);
  }
  else
  {
    fallback_log(x, y, n);
  }
  return VMATH_OK;
}

int vmath_log10(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_log10(x, y, n);
  }
  else
  {
    fallback_log10(x, y, n);
  }
  return VMATH_OK;
}

int vmath_sin(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_sin(x, y, n);
  }
  else
  {
    fallback_sin(x, y, n);
  }
  return VMATH_OK;
}

int vmath_cos(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_cos(x, y, n);
  }
  else
  {
    fallback_cos(x, y, n);
  }
  return VMATH_OK;
}

int vmath_tan(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_tan(x, y, n);
  }
  else
  {
    fallback_tan(x, y, n);
  }
  return VMATH_OK;
}

int vmath_asin(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_asin(x, y, n);
  return VMATH_OK;
}

int vmath_acos(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_acos(x, y, n);
  return VMATH_OK;
}

int vmath_atan(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_atan(x, y, n);
  return VMATH_OK;
}

int vmath_atan2(const double *y_in, const double *x, double *result, size_t n)
{
  if (!g_vmath_initialized)
    return VMATH_ERR_NOT_INITIALIZED;
  if (!y_in || !x || !result || n == 0)
    return VMATH_ERR_INVALID_ARG;

  sleef_vec_atan2(y_in, x, result, n);
  return VMATH_OK;
}

int vmath_sinh(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_sinh(x, y, n);
  return VMATH_OK;
}

int vmath_cosh(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_cosh(x, y, n);
  return VMATH_OK;
}

int vmath_tanh(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_tanh(x, y, n);
  return VMATH_OK;
}

int vmath_sqrt(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_sqrt(x, y, n);
  }
  else
  {
    fallback_sqrt(x, y, n);
  }
  return VMATH_OK;
}

int vmath_cbrt(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_cbrt(x, y, n);
  return VMATH_OK;
}

int vmath_pow_scalar(const double *x, double p, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  fallback_pow_scalar(x, p, y, n);
  return VMATH_OK;
}

int vmath_pow(const double *x, const double *y_in, double *result, size_t n)
{
  if (!g_vmath_initialized)
    return VMATH_ERR_NOT_INITIALIZED;
  if (!x || !y_in || !result || n == 0)
    return VMATH_ERR_INVALID_ARG;

  if (g_current_isa >= VMATH_ISA_AVX2 && sleef_is_available())
  {
    sleef_vec_pow(x, y_in, result, n);
  }
  else
  {
    fallback_pow(x, y_in, result, n);
  }
  return VMATH_OK;
}

int vmath_erf(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_erf(x, y, n);
  return VMATH_OK;
}

int vmath_erfc(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  sleef_vec_erfc(x, y, n);
  return VMATH_OK;
}

int vmath_gamma(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  fallback_gamma(x, y, n);
  return VMATH_OK;
}

int vmath_lgamma(const double *x, double *y, size_t n)
{
  int ret = validate_inputs(x, y, n);
  if (ret != VMATH_OK)
    return ret;

  fallback_lgamma(x, y, n);
  return VMATH_OK;
}

// ============================================================================
// Diagnostic & Performance
// ============================================================================

const char *vmath_get_info(void)
{
  static char info[256];
  snprintf(info, sizeof(info),
           "Vector Math Engine - ISA: %s, SLEEF: %s",
           vmath_isa_name(g_current_isa),
           sleef_is_available() ? "Available" : "Not Available");
  return info;
}

int vmath_self_test(void)
{
  if (!g_vmath_initialized)
  {
    return VMATH_ERR_NOT_INITIALIZED;
  }

  // Basic correctness test
  const size_t n = 8;
  float x[8] = {0.0f, 0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 3.14159f, -3.14159f};
  float y[8];

  // Test exp
  vmath_expf(x, y, n);
  for (size_t i = 0; i < n; ++i)
  {
    float expected = std::exp(x[i]);
    float diff = std::fabs(y[i] - expected);
    if (diff > 1e-5f)
    {
      fprintf(stderr, "[VMATH] Self-test failed: exp(%f) = %f, expected %f\n",
              x[i], y[i], expected);
      return VMATH_ERR_INTERNAL;
    }
  }

  printf("[VMATH] Self-test passed\n");
  return VMATH_OK;
}