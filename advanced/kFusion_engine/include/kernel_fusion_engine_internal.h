// advanced/kFusion_engine/include/kernel_fusion_engine_internal.h
#ifndef KERNEL_FUSION_ENGINE_INTERNAL_H_
#define KERNEL_FUSION_ENGINE_INTERNAL_H_

#include "kernel_fusion_engine.h"

// Derived component headers
#include "microkernel_interface.h"
#include "pool_manager.h"

// Base component headers
#include "ffm_cache_block.h"
#include "thread_scheduler.h"

// Standard library headers
#include <algorithm>
#include <atomic>
#include <cmath>
#include <immintrin.h> // AVX/AVX2/AVX-512 intrinsics
#include <mutex>

#include "global_thread_scheduler.h"

namespace kfe_internal {

/* ========================================================================== */
/* Global State Structure                                                     */
/* ========================================================================== */

struct KFEState {
  bool initialized;
  kfe_config_t config;
  pm_t *pool_manager;
  jcore::global_thread::GlobalThreadScheduler *scheduler;
  ffm_cache_info_t *cache_info;
  std::mutex state_mutex;
  std::atomic<size_t> total_fused_ops{0};
  std::atomic<size_t> total_memory_saved{0};

  KFEState()
    : initialized(false), config(), pool_manager(nullptr), scheduler(nullptr),
      cache_info(nullptr) {
  }
};

// Global state instance (defined in kernel_fusion_engine_core.cpp)
extern KFEState g_kfe_state;

/* ========================================================================== */
/* Scalar Activation Functions                                                */
/* ========================================================================== */

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
inline float relu6(float x) { return std::min(std::max(0.0f, x), 6.0f); }
inline float tanh_act(float x) { return std::tanh(x); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float leaky_relu(float x) { return x > 0.0f ? x : 0.01f * x; }
inline float swish(float x) { return x * sigmoid(x); }

inline float gelu(float x) {
  const float sqrt_2_over_pi = 0.7978845608f;
  const float coeff = 0.044715f;
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + coeff * x3);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

/* ========================================================================== */
/* Vectorized Activation Functions (AVX2)                                     */
/* ========================================================================== */

#ifdef __AVX2__
__m256 avx2_relu(__m256 x);
__m256 avx2_relu6(__m256 x);
__m256 avx2_leaky_relu(__m256 x);
__m256 avx2_sigmoid(__m256 x);
#endif

/* ========================================================================== */
/* Helper Function Declarations                                               */
/* ========================================================================== */

  /**
 * @brief Apply activation function (scalar fallback)
 */
void apply_activation_scalar(float *data, size_t n, kfe_activation_t act);

  /**
 * @brief Apply activation function (vectorized using SIMD)
 */
void apply_activation_vectorized(float *data, size_t n, kfe_activation_t act);

  /**
   * @brief Add bias to row-major matrix: C[i,j] += bias[j]
   */
void add_bias_row_major(float *C, size_t m, size_t n, size_t ldc,
                        const float *bias);

  /**
  * @brief Add bias to column-major matrix: C[i,j] += bias[j]
  */
void add_bias_column_major(float *C, size_t m, size_t n, size_t ldc, const float *bias);

  /**
 * @brief Elementwise addition: C = A + beta * D
 */
void elementwise_add(float *C, size_t m, size_t n, size_t ldc, const float *D,
                     size_t ldd, float beta);

  /**
 * @brief Elementwise multiplication: C = A * B (vectorized)
 */

void elementwise_mul(float *C, size_t m, size_t n, size_t ldc,
                     const float *D, size_t ldd, float beta);

inline mil_layout_t to_mil_layout(kfe_layout_t layout) {
  return (layout == KFE_LAYOUT_ROW_MAJOR) ? MIL_LAYOUT_ROW_MAJOR
                                          : MIL_LAYOUT_COL_MAJOR;
}

inline mil_transpose_t to_mil_transpose(kfe_transpose_t trans) {
  return (trans == KFE_NO_TRANS) ? MIL_NO_TRANS : MIL_TRANS;
}

} // namespace kfe_internal

#endif // KERNEL_FUSION_ENGINE_INTERNAL_H_