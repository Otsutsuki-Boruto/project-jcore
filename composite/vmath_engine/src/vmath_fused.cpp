// vmath_fused.cpp
// Fused ML/scientific operations (sigmoid, relu, gelu, softplus)

#include "vmath_engine.h"
#include "vmath_sleef_wrapper.h"
#include "vmath_fallback.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// ============================================================================
// Sigmoid: 1 / (1 + exp(-x))
// ============================================================================

int vmath_sigmoidf(const float *x, float *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  // Step 1: Negate the input array in-place or via temporary
  std::vector<float> neg_x(n);
  for (size_t i = 0; i < n; ++i) {
    neg_x[i] = -x[i];
  }

  // Step 2: Compute vectorized exponential using SLEEF
  sleef_vec_expf(neg_x.data(), y, n);

  // Step 3: Compute final sigmoid: 1 / (1 + exp(-x))
  for (size_t i = 0; i < n; ++i) {
    y[i] = 1.0f / (1.0f + y[i]);
  }

  return VMATH_OK;
}


int vmath_sigmoid(const double *x, double *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  // Step 1: Prepare negated input
  std::vector<double> neg_x(n);
  for (size_t i = 0; i < n; ++i) {
    neg_x[i] = -x[i];
  }

  // Step 2: Compute vectorized exponential using SLEEF
  sleef_vec_exp(neg_x.data(), y, n);

  // Step 3: Compute sigmoid
  for (size_t i = 0; i < n; ++i) {
    y[i] = 1.0 / (1.0 + y[i]);
  }

  return VMATH_OK;
}

// ============================================================================
// ReLU: max(0, x)
// ============================================================================

int vmath_reluf(const float *x, float *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  // ReLU: max(x, 0) - we need an array of zeros
  std::vector<float> zeros(n, 0.0f);
  sleef_vec_maxf(x, zeros.data(), y, n);

  return VMATH_OK;
}

int vmath_relu(const double *x, double *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  // ReLU: max(x, 0) - we need an array of zeros
  std::vector<double> zeros(n, 0.0);
  sleef_vec_max(x, zeros.data(), y, n);

  return VMATH_OK;
}
// ============================================================================
// ReLU6
// ============================================================================

int vmath_relu6f(const float *x, float *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  std::vector<float> zeros(n, 0.0f);
  std::vector<float> sixes(n, 6.0f);
  std::vector<float> temp(n);

  // Step 1: max(x, 0)
  sleef_vec_maxf(x, zeros.data(), temp.data(), n);

  // Step 2: min(temp, 6)
  sleef_vec_minf(temp.data(), sixes.data(), y, n);

  return VMATH_OK;
}

int vmath_relu6(const double *x, double *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  std::vector<double> zeros(n, 0.0);
  std::vector<double> sixes(n, 6.0);
  std::vector<double> temp(n);

  // Step 1: max(x, 0)
  sleef_vec_max(x, zeros.data(), temp.data(), n);

  // Step 2: min(temp, 6)
  sleef_vec_min(temp.data(), sixes.data(), y, n);

  return VMATH_OK;
}

// ============================================================================
// Leaky ReLU
// ============================================================================

int vmath_leaky_reluf(const float *x, float *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  const float alpha = 0.01f;
  std::vector<float> alpha_x(n);

  // Compute alpha * x
  for (size_t i = 0; i < n; ++i) {
    alpha_x[i] = alpha * x[i];
  }

  // Select: max(x, alpha*x)
  sleef_vec_maxf(x, alpha_x.data(), y, n);

  return VMATH_OK;
}

int vmath_leaky_relu(const double *x, double *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  const double alpha = 0.01;
  std::vector<double> alpha_x(n);

  // Compute alpha * x
  for (size_t i = 0; i < n; ++i) {
    alpha_x[i] = alpha * x[i];
  }

  // Select: max(x, alpha*x)
  sleef_vec_max(x, alpha_x.data(), y, n);

  return VMATH_OK;
}

// ============================================================================
// Softplus: log(1 + exp(x))
// ============================================================================

int vmath_softplusf(const float *x, float *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  // Step 1: Compute vectorized exponential of x
  std::vector<float> exp_x(n);
  sleef_vec_expf(x, exp_x.data(), n);

  // Step 2: Compute vectorized log1p of exp_x for Softplus
  sleef_vec_log1pf(exp_x.data(), y, n);

  return VMATH_OK;
}

int vmath_softplus(const double *x, double *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  // Step 1: Compute vectorized exponential of x
  std::vector<double> exp_x(n);
  sleef_vec_exp(x, exp_x.data(), n);

  // Step 2: Compute vectorized log1p of exp_x for Softplus
  sleef_vec_log1p(exp_x.data(), y, n);

  return VMATH_OK;
}


// ============================================================================
// GELU: x * Φ(x) where Φ is standard normal CDF
// Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
// ============================================================================

int vmath_geluf(const float *x, float *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
  const float coeff = 0.044715f;

  // Step 1: Compute x^3 using vectorized power
  std::vector<float> x3(n);
  std::vector<float> exponent(n, 3.0f); // exponent array filled with 3.0
  sleef_vec_powf(x, exponent.data(), x3.data(), n);

  // Step 2: Compute inner = sqrt(2/pi) * (x + coeff * x^3)
  std::vector<float> inner(n);
  for (size_t i = 0; i < n; ++i) {
    inner[i] = sqrt_2_over_pi * (x[i] + coeff * x3[i]);
  }

  // Step 3: Vectorized tanh
  std::vector<float> tanh_inner(n);
  sleef_vec_tanhf(inner.data(), tanh_inner.data(), n);

  // Step 4: Compute GELU output
  for (size_t i = 0; i < n; ++i) {
    y[i] = 0.5f * x[i] * (1.0f + tanh_inner[i]);
  }

  return VMATH_OK;
}

int vmath_gelu(const double *x, double *y, size_t n)
{
  if (!x || !y || n == 0) {
    return VMATH_ERR_INVALID_ARG;
  }

  const double sqrt_2_over_pi = 0.7978845608028654;
  const double coeff = 0.044715;

  // Step 1: Compute x^3 using vectorized power
  std::vector<double> x3(n);
  std::vector<double> exponent(n, 3.0); // array filled with 3.0 for power
  sleef_vec_pow(x, exponent.data(), x3.data(), n);

  // Step 2: Compute inner = sqrt(2/pi) * (x + coeff * x^3)
  std::vector<double> inner(n);
  for (size_t i = 0; i < n; ++i) {
    inner[i] = sqrt_2_over_pi * (x[i] + coeff * x3[i]);
  }

  // Step 3: Vectorized tanh
  std::vector<double> tanh_inner(n);
  sleef_vec_tanh(inner.data(), tanh_inner.data(), n);

  // Step 4: Compute GELU output
  for (size_t i = 0; i < n; ++i) {
    y[i] = 0.5 * x[i] * (1.0 + tanh_inner[i]);
  }

  return VMATH_OK;
}
