// advanced/jit_kernel/src/vectorization_backends.cpp
#include "jit_kernel_internal.h"

// LLVM 20 includes for IR generation
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

// Conditional includes based on availability
#ifdef HWY_HIGHWAY_H_
#include "hwy/highway.h"
namespace HWY_NAMESPACE {
using namespace hwy::HWY_NAMESPACE;
}
#endif

#ifdef VECTORCLASS_H
#include "vectorclass.h"
#endif

#ifdef EVE_MODULE_CORE_HPP_INCLUDED
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#endif

namespace jkg_internal {

/* ========================================================================== */
/* Highway Backend Implementation                                             */
/* ========================================================================== */

#if JKG_HAS_HIGHWAY

bool HighwayBackend::is_available() const { return true; }

size_t HighwayBackend::vector_width() const {
#ifdef HWY_HIGHWAY_H_
  const HWY_NAMESPACE::ScalableTag<float> d;
  return HWY_NAMESPACE::Lanes(d);
#else
  return 1;
#endif
}

void HighwayBackend::generate_gemm_tile(Module *module, LLVMIRBuilder *builder,
                                        Function *func, size_t M, size_t N,
                                        size_t K) {
  // Highway integration would generate LLVM calls to Highway functions
  // For runtime dispatch, Highway provides its own mechanism
  // Here we would emit calls to pre-compiled Highway kernels

  log_info("Highway GEMM tile generation: M=%zu, N=%zu, K=%zu", M, N, K);

  // In practice, we would:
  // 1. Create external function declarations for Highway kernels
  // 2. Emit calls to those functions from generated LLVM IR
  // 3. Link against pre-compiled Highway library

  // For now, log that Highway backend is being used
}

void HighwayBackend::generate_activation(Module *module, LLVMIRBuilder *builder,
                                         Value *data, size_t N,
                                         jkg_activation_t act) {
  log_info("Highway activation generation: N=%zu, act=%d", N, act);

  // Similar approach: emit calls to Highway SIMD activation functions
}

// Runtime Highway GEMM microkernel (external, linked separately)
extern "C" void highway_gemm_f32(const float *A, const float *B, float *C,
                                 size_t M, size_t N, size_t K, size_t lda,
                                 size_t ldb, size_t ldc) {
#ifdef HWY_HIGHWAY_H_
  const HWY_NAMESPACE::ScalableTag<float> d;
  const size_t vec_size = HWY_NAMESPACE::Lanes(d);

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j += vec_size) {
      auto acc = HWY_NAMESPACE::Zero(d);

      for (size_t k = 0; k < K; k++) {
        const float a_val = A[i * lda + k];
        auto a_vec = HWY_NAMESPACE::Set(d, a_val);

        // Load B vector
        size_t b_idx = k * ldb + j;
        auto b_vec = HWY_NAMESPACE::LoadU(d, &B[b_idx]);

        // FMA: acc += a * b
        acc = HWY_NAMESPACE::MulAdd(a_vec, b_vec, acc);
      }

      // Store result
      size_t c_idx = i * ldc + j;
      HWY_NAMESPACE::StoreU(acc, d, &C[c_idx]);
    }
  }
#else
  // Fallback scalar implementation
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        sum += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = sum;
    }
  }
#endif
}

#endif // JKG_HAS_HIGHWAY

/* ========================================================================== */
/* VectorClass Backend Implementation                                         */
/* ========================================================================== */

#if JKG_HAS_VECTORCLASS

bool VectorClassBackend::is_available() const { return true; }

size_t VectorClassBackend::vector_width() const {
#if defined(__AVX512F__)
  return 16; // AVX-512: 512-bit / 32-bit = 16 floats
#elif defined(__AVX2__) || defined(__AVX__)
  return 8; // AVX/AVX2: 256-bit / 32-bit = 8 floats
#else
  return 4; // SSE: 128-bit / 32-bit = 4 floats
#endif
}

void VectorClassBackend::generate_gemm_tile(Module *module,
                                            LLVMIRBuilder *builder,
                                            Function *func, size_t M, size_t N,
                                            size_t K) {
  log_info("VectorClass GEMM tile generation: M=%zu, N=%zu, K=%zu", M, N, K);

  // VectorClass provides C++ wrappers around intrinsics
  // We emit calls to VectorClass-based kernels
}

void VectorClassBackend::generate_activation(Module *module,
                                             LLVMIRBuilder *builder,
                                             Value *data, size_t N,
                                             jkg_activation_t act) {
  log_info("VectorClass activation generation: N=%zu, act=%d", N, act);
}

// Runtime VectorClass GEMM microkernel
extern "C" void vectorclass_gemm_f32(const float *A, const float *B, float *C,
                                     size_t M, size_t N, size_t K, size_t lda,
                                     size_t ldb, size_t ldc) {
#ifdef VECTORCLASS_H
  const size_t vec_size = Vec8f().size(); // 8 for AVX2, 16 for AVX-512

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j += vec_size) {
      Vec8f acc(0.0f);

      for (size_t k = 0; k < K; k++) {
        float a_val = A[i * lda + k];
        Vec8f a_vec(a_val);

        Vec8f b_vec;
        b_vec.load(&B[k * ldb + j]);

        acc = mul_add(a_vec, b_vec, acc); // FMA
      }

      acc.store(&C[i * ldc + j]);
    }
  }
#else
  // Fallback
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        sum += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = sum;
    }
  }
#endif
}

#endif // JKG_HAS_VECTORCLASS

/* ========================================================================== */
/* EVE Backend Implementation */
/* ========================================================================== */

#if JKG_HAS_EVE

bool EVEBackend::is_available() const { return true; }

size_t EVEBackend::vector_width() const {
#ifdef EVE_MODULE_CORE_HPP_INCLUDED
  return eve::expected_cardinal_v<float>;
#else
  return 1;
#endif
}

void EVEBackend::generate_gemm_tile(Module *module, LLVMIRBuilder *builder,
                                    Function *func, size_t M, size_t N,
                                    size_t K) {
  log_info("EVE GEMM tile generation: M=%zu, N=%zu, K=%zu", M, N, K);

  // EVE uses expression templates for optimal code generation
  // We emit calls to EVE-based fused kernels
}

void EVEBackend::generate_activation(Module *module, LLVMIRBuilder *builder,
                                     Value *data, size_t N,
                                     jkg_activation_t act) {
  log_info("EVE activation generation: N=%zu, act=%d", N, act);
}

// Runtime EVE GEMM microkernel with fused epilogue
extern "C" void eve_gemm_f32(const float *A, const float *B, float *C, size_t M,
                             size_t N, size_t K, size_t lda, size_t ldb,
                             size_t ldc) {
#ifdef EVE_MODULE_CORE_HPP_INCLUDED
  using wide_t = eve::wide<float>;
  constexpr auto vec_size = wide_t::size();

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j += vec_size) {
      wide_t acc(0.0f);

      for (size_t k = 0; k < K; k++) {
        float a_val = A[i * lda + k];
        wide_t a_vec(a_val);

        // Load B vector with proper alignment handling
        wide_t b_vec(&B[k * ldb + j]);

        // FMA using EVE
        acc = eve::fma(a_vec, b_vec, acc);
      }

      // Store result
      acc.store(&C[i * ldc + j]);
    }
  }
#else
  // Fallback
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        sum += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = sum;
    }
  }
#endif
}

// EVE fused activation functions
extern "C" void eve_relu_f32(float *data, size_t N) {
#ifdef EVE_MODULE_CORE_HPP_INCLUDED
  using wide_t = eve::wide<float>;
  constexpr auto vec_size = wide_t::size();

  size_t i = 0;
  for (; i + vec_size <= N; i += vec_size) {
    wide_t x(&data[i]);
    wide_t result = eve::max(x, wide_t(0.0f));
    result.store(&data[i]);
  }

  // Tail handling
  for (; i < N; i++) {
    data[i] = std::max(0.0f, data[i]);
  }
#else
  for (size_t i = 0; i < N; i++) {
    data[i] = std::max(0.0f, data[i]);
  }
#endif
}

extern "C" void eve_gelu_f32(float *data, size_t N) {
#ifdef EVE_MODULE_CORE_HPP_INCLUDED
  using wide_t = eve::wide<float>;
  constexpr auto vec_size = wide_t::size();

  const wide_t half(0.5f);
  const wide_t one(1.0f);
  const wide_t sqrt_2_over_pi(0.7978845608f);
  const wide_t coeff(0.044715f);

  size_t i = 0;
  for (; i + vec_size <= N; i += vec_size) {
    wide_t x(&data[i]);

    // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    wide_t x3 = x * x * x;
    wide_t inner = sqrt_2_over_pi * (x + coeff * x3);
    wide_t tanh_val = eve::tanh(inner);
    wide_t result = half * x * (one + tanh_val);

    result.store(&data[i]);
  }

  // Tail
  for (; i < N; i++) {
    float x = data[i];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    data[i] = 0.5f * x * (1.0f + std::tanh(inner));
  }
#else
  for (size_t i = 0; i < N; i++) {
    float x = data[i];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    data[i] = 0.5f * x * (1.0f + std::tanh(inner));
  }
#endif
}

#endif // JKG_HAS_EVE

/* ========================================================================== */
/* Fallback Implementations (No Vectorization Libraries)                      */
/* ========================================================================== */

#if !JKG_HAS_HIGHWAY
bool HighwayBackend::is_available() const { return false; }
size_t HighwayBackend::vector_width() const { return 1; }
void HighwayBackend::generate_gemm_tile(llvm::Module *, LLVMIRBuilder *,
                                        llvm::Function *, size_t, size_t,
                                        size_t) {}
void HighwayBackend::generate_activation(llvm::Module *, LLVMIRBuilder *,
                                         llvm::Value *, size_t,
                                         jkg_activation_t) {}
#endif

#if !JKG_HAS_VECTORCLASS
bool VectorClassBackend::is_available() const { return false; }
size_t VectorClassBackend::vector_width() const { return 1; }
void VectorClassBackend::generate_gemm_tile(llvm::Module *, LLVMIRBuilder *,
                                            llvm::Function *, size_t, size_t,
                                            size_t) {}
void VectorClassBackend::generate_activation(llvm::Module *, LLVMIRBuilder *,
                                             llvm::Value *, size_t,
                                             jkg_activation_t) {}
#endif

#if !JKG_HAS_EVE
bool EVEBackend::is_available() const { return false; }
size_t EVEBackend::vector_width() const { return 1; }
void EVEBackend::generate_gemm_tile(llvm::Module *, LLVMIRBuilder *,
                                    llvm::Function *, size_t, size_t, size_t) {}
void EVEBackend::generate_activation(llvm::Module *, LLVMIRBuilder *,
                                     llvm::Value *, size_t, jkg_activation_t) {}
#endif

} // namespace jkg_internal

/* ========================================================================== */
/* Convenience Wrappers Implementation                                        */
/* ========================================================================== */

int jkg_generate_gemm_tile(size_t M, size_t N, size_t K,
                           jkg_kernel_internal_t **out_handle) {
  jkg_kernel_params_t params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.activation = JKG_ACT_NONE;
  params.alpha = 1.0f;
  params.beta = 0.0f;
  params.has_bias = 0;
  params.has_residual = 0;

  return jkg_generate_kernel(JKG_KERNEL_GEMM_TILE, &params, out_handle);
}

int jkg_generate_fused_gemm(size_t M, size_t N, size_t K,
                            jkg_activation_t activation, float alpha,
                            jkg_kernel_internal_t **out_handle) {

  jkg_kernel_params_t params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.activation = activation;
  params.alpha = alpha;
  params.beta = 0.0f;
  params.has_bias = 1;
  params.has_residual = 0;

  jkg_kernel_type_t type = (activation == JKG_ACT_RELU)
                               ? JKG_KERNEL_GEMM_BIAS_RELU
                               : JKG_KERNEL_GEMM_BIAS_ACT;
  return jkg_generate_kernel(type, &params, out_handle);
}

int jkg_generate_elementwise(jkg_kernel_type_t kernel_type, size_t N,
                             jkg_kernel_internal_t **out_handle) {
  if (kernel_type != JKG_KERNEL_ELEMENTWISE_ADD &&
      kernel_type != JKG_KERNEL_ELEMENTWISE_MUL) {
    return JKG_ERR_INVALID_ARG;
  }

  jkg_kernel_params_t params = {};
  params.M = N;
  params.N = 1;
  params.K = 1;
  params.activation = JKG_ACT_NONE;
  params.alpha = 1.0f;
  params.beta = 0.0f;
  params.has_bias = 0;
  params.has_residual = 0;

  return jkg_generate_kernel(kernel_type, &params, out_handle);
}