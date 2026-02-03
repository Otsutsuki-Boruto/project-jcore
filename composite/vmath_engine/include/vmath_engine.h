#ifndef VMATH_ENGINE_H
#define VMATH_ENGINE_H

/**
 * @file vmath_engine.h
 * @brief Vector Math Engine - SIMD-optimized transcendental operations
 *
 * Derived Component: Vector Math Engine (SLEEF/xsimd Integration)
 * Purpose: Vectorized transcendental operations (exp, log, sin, cos, tan, etc.)
 * Dependencies:
 *   - Kernel Dispatch Table/Runtime Selector (derived)
 *   - CPU Feature Detection Module (base)
 *   - ISA-Aware Dispatch Mechanism (base)
 *
 * Design:
 *   - Runtime ISA selection (scalar, SSE2, AVX, AVX2, AVX-512)
 *   - SLEEF library integration for high-precision transcendentals
 *   - Fallback scalar implementations
 *   - FFM-compatible C API
 *   - Google C++ style, modular design
 *
 * Thread-safety: All functions are thread-safe (stateless).
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ========================================================================== */
  /* Error Codes                                                                 */
  /* ========================================================================== */

#define VMATH_OK 0
#define VMATH_ERR_NOT_INITIALIZED -1
#define VMATH_ERR_INVALID_ARG -2
#define VMATH_ERR_NO_SIMD_SUPPORT -3
#define VMATH_ERR_INTERNAL -4

  /* ========================================================================== */
  /* ISA Feature Levels                                                          */
  /* ========================================================================== */

  typedef enum
  {
    VMATH_ISA_SCALAR = 0, /**< Portable C fallback */
    VMATH_ISA_SSE2,       /**< SSE2 (baseline x86_64) */
    VMATH_ISA_AVX,        /**< AVX */
    VMATH_ISA_AVX2,       /**< AVX2 + FMA */
    VMATH_ISA_AVX512      /**< AVX-512F */
  } vmath_isa_level_t;

  /* ========================================================================== */
  /* Initialization & Query                                                      */
  /* ========================================================================== */

  /**
   * @brief Initialize the Vector Math Engine
   *
   * Detects CPU features and registers appropriate SIMD kernels.
   * Must be called before any vmath operations.
   *
   * @return VMATH_OK on success, error code otherwise
   */
  int vmath_init(void);

  /**
   * @brief Shutdown the Vector Math Engine
   *
   * Cleanup internal state. Safe to call multiple times.
   */
  void vmath_shutdown(void);

  /**
   * @brief Query the current active ISA level
   *
   * @return Current ISA level in use
   */
  vmath_isa_level_t vmath_get_isa_level(void);

  /**
   * @brief Get human-readable ISA name
   *
   * @param level ISA level enum
   * @return String name (e.g., "AVX2")
   */
  const char *vmath_isa_name(vmath_isa_level_t level);

  /* ========================================================================== */
  /* Vector Operations (Single Precision - float)                               */
  /* ========================================================================== */

  /**
   * @brief Compute exp(x) for vector of floats
   * @param x Input array
   * @param y Output array (can alias x for in-place)
   * @param n Number of elements
   * @return VMATH_OK or error code
   */
  int vmath_expf(const float *x, float *y, size_t n);

  /**
   * @brief Compute natural log ln(x) for vector of floats
   * @param x Input array (must be > 0)
   * @param y Output array
   * @param n Number of elements
   * @return VMATH_OK or error code
   */
  int vmath_logf(const float *x, float *y, size_t n);

  /**
   * @brief Compute log10(x) for vector of floats
   */
  int vmath_log10f(const float *x, float *y, size_t n);

  /**
   * @brief Compute sine sin(x) for vector of floats
   * @param x Input array (radians)
   * @param y Output array
   * @param n Number of elements
   */
  int vmath_sinf(const float *x, float *y, size_t n);

  /**
   * @brief Compute cosine cos(x)
   */
  int vmath_cosf(const float *x, float *y, size_t n);

  /**
   * @brief Compute tangent tan(x)
   */
  int vmath_tanf(const float *x, float *y, size_t n);

  /**
   * @brief Compute arc sine asin(x) (domain [-1, 1])
   */
  int vmath_asinf(const float *x, float *y, size_t n);

  /**
   * @brief Compute arc cosine acos(x)
   */
  int vmath_acosf(const float *x, float *y, size_t n);

  /**
   * @brief Compute arc tangent atan(x)
   */
  int vmath_atanf(const float *x, float *y, size_t n);

  /**
   * @brief Compute two-argument arc tangent atan2(y, x)
   * @param y Numerator array
   * @param x Denominator array
   * @param result Output array
   * @param n Number of elements
   */
  int vmath_atan2f(const float *y, const float *x, float *result, size_t n);

  /**
   * @brief Compute hyperbolic sine sinh(x)
   */
  int vmath_sinhf(const float *x, float *y, size_t n);

  /**
   * @brief Compute hyperbolic cosine cosh(x)
   */
  int vmath_coshf(const float *x, float *y, size_t n);

  /**
   * @brief Compute hyperbolic tangent tanh(x)
   */
  int vmath_tanhf(const float *x, float *y, size_t n);

  /**
   * @brief Compute square root sqrt(x)
   */
  int vmath_sqrtf(const float *x, float *y, size_t n);

  /**
   * @brief Compute cube root cbrt(x)
   */
  int vmath_cbrtf(const float *x, float *y, size_t n);

  /**
   * @brief Compute power x^p (scalar exponent)
   * @param x Base array
   * @param p Exponent (scalar)
   * @param y Output array
   * @param n Number of elements
   */
  int vmath_powf_scalar(const float *x, float p, float *y, size_t n);

  /**
   * @brief Compute power x^y (vector exponent)
   * @param x Base array
   * @param y Exponent array
   * @param result Output array
   * @param n Number of elements
   */
  int vmath_powf(const float *x, const float *y, float *result, size_t n);

  /**
   * @brief Compute error function erf(x)
   */
  int vmath_erff(const float *x, float *y, size_t n);

  /**
   * @brief Compute complementary error function erfc(x)
   */
  int vmath_erfcf(const float *x, float *y, size_t n);

  /**
   * @brief Compute gamma function Γ(x)
   */
  int vmath_gammaf(const float *x, float *y, size_t n);

  /**
   * @brief Compute natural log of gamma function ln(Γ(x))
   */
  int vmath_lgammaf(const float *x, float *y, size_t n);

  /* ========================================================================== */
  /* Vector Operations (Double Precision - double)                              */
  /* ========================================================================== */

  int vmath_exp(const double *x, double *y, size_t n);
  int vmath_log(const double *x, double *y, size_t n);
  int vmath_log10(const double *x, double *y, size_t n);
  int vmath_sin(const double *x, double *y, size_t n);
  int vmath_cos(const double *x, double *y, size_t n);
  int vmath_tan(const double *x, double *y, size_t n);
  int vmath_asin(const double *x, double *y, size_t n);
  int vmath_acos(const double *x, double *y, size_t n);
  int vmath_atan(const double *x, double *y, size_t n);
  int vmath_atan2(const double *y, const double *x, double *result, size_t n);
  int vmath_sinh(const double *x, double *y, size_t n);
  int vmath_cosh(const double *x, double *y, size_t n);
  int vmath_tanh(const double *x, double *y, size_t n);
  int vmath_sqrt(const double *x, double *y, size_t n);
  int vmath_cbrt(const double *x, double *y, size_t n);
  int vmath_pow_scalar(const double *x, double p, double *y, size_t n);
  int vmath_pow(const double *x, const double *y, double *result, size_t n);
  int vmath_erf(const double *x, double *y, size_t n);
  int vmath_erfc(const double *x, double *y, size_t n);
  int vmath_gamma(const double *x, double *y, size_t n);
  int vmath_lgamma(const double *x, double *y, size_t n);

  /* ========================================================================== */
  /* Fused Operations (Common ML/Scientific patterns)                           */
  /* ========================================================================== */

  /**
   * @brief Compute sigmoid: 1 / (1 + exp(-x))
   */
  int vmath_sigmoidf(const float *x, float *y, size_t n);
  int vmath_sigmoid(const double *x, double *y, size_t n);

  /**
   * @brief Compute ReLU: max(0, x)
   */
  int vmath_reluf(const float *x, float *y, size_t n);
  int vmath_relu(const double *x, double *y, size_t n);

 /**
  * @brief Compute Leaky ReLU: x if x > 0, else alpha*x (alpha=0.01)
  */
  int vmath_leaky_reluf(const float *x, float *y, size_t n);
  int vmath_leaky_relu(const double *x, double *y, size_t n);

  /**
   * @brief Compute ReLU6: min(max(x, 0), 6)
   */
  int vmath_relu6f(const float *x, float *y, size_t n);
  int vmath_relu6(const double *x, double *y, size_t n);

  /**
   * @brief Compute softplus: log(1 + exp(x))
   */
  int vmath_softplusf(const float *x, float *y, size_t n);
  int vmath_softplus(const double *x, double *y, size_t n);

  /**
   * @brief Compute GELU: x * Φ(x) where Φ is standard normal CDF
   * Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
   */
  int vmath_geluf(const float *x, float *y, size_t n);
  int vmath_gelu(const double *x, double *y, size_t n);

  /* ========================================================================== */
  /* Diagnostic & Performance                                                    */
  /* ========================================================================== */

  /**
   * @brief Get performance info string (human-readable)
   * @return Static string with ISA level and feature info
   */
  const char *vmath_get_info(void);

  /**
   * @brief Run internal correctness self-test
   * @return VMATH_OK if all tests pass
   */
  int vmath_self_test(void);

#ifdef __cplusplus
}
#endif

#endif /* VMATH_ENGINE_H */