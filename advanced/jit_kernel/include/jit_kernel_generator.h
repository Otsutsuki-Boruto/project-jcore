// advanced/jit_kernel/include/jit_kernel_generator.h
#ifndef JCORE_JIT_KERNEL_GENERATOR_H_
#define JCORE_JIT_KERNEL_GENERATOR_H_

/**
 * @file jit_kernel_generator.h
 * @brief JIT Kernel Generator - Runtime Code Generation for Fused Operations
 *
 * Component: JIT Kernel Generator (Advanced)
 * Purpose: Generates optimized fused kernels at runtime using LLVM 20 ORC
 *
 * Dependencies:
 *   Advanced:
 *     - Kernel Fusion Engine: Fusion patterns and epilogue operations
 *   Derived:
 *     - Adaptive Kernel Auto-Tuner: Performance profiling and kernel selection
 *   Base:
 *     - CPU Feature Detection Module: ISA capabilities
 *     - ISA-Aware Dispatch: Runtime dispatch mechanism
 *     - Hardware Introspection Layer: Topology and cache info
 *     - Microbenchmark & Timer Utilities: Performance measurement
 *
 * Vectorization Libraries:
 *   - Google Highway: Portable SIMD with runtime dispatch
 *   - VectorClass: x86-specific intrinsics for maximum performance
 *   - EVE: Expressive vector engine for complex math sequences (C++20
 * compatible)
 *
 * Features:
 *   - Runtime kernel generation and compilation (LLVM 20 ORC JIT)
 *   - Automatic ISA selection (AVX2/AVX-512/AMX)
 *   - Fused operation chains (GEMM + Bias + Activation)
 *   - Custom microkernels with optimal tile sizes
 *   - LLVM IR generation and optimization
 *   - Code caching for repeated compilations
 *   - C++17/20 compatible
 *
 * Requirements:
 *   - LLVM 20.x
 *   - C++17 or C++20 compiler
 *   - OpenBLAS/BLIS for benchmarking
 *
 * Thread-safety: Thread-safe after initialization
 * FFM API: Fully compatible with Project JCore FFM API
 */

#include <cstddef>
#include <cstdint>

/* Export macro for shared library */
#ifdef _WIN32
#ifdef JCORE_BUILD_DLL
#define JCORE_API __declspec(dllexport)
#else
#define JCORE_API __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define JCORE_API __attribute__((visibility("default")))
#else
#define JCORE_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque kernel handle type - forward declaration only, no typedef yet */
struct jkg_kernel_internal_t;

/* ========================================================================== */
/* Error Codes */
/* ========================================================================== */

#define JKG_OK 0
#define JKG_ERR_NOT_INITIALIZED -1
#define JKG_ERR_INVALID_ARG -2
#define JKG_ERR_NO_MEMORY -3
#define JKG_ERR_INTERNAL -4
#define JKG_ERR_COMPILATION -5
#define JKG_ERR_UNSUPPORTED_ISA -6
#define JKG_ERR_CACHE_MISS -7
#define JKG_ERR_LLVM_INIT -8
#define JKG_ERR_NOT_FOUND -9
#define JKG_ERR_IR_INVALID -10

/* ========================================================================== */
/* ISA Target Flags */
/* ========================================================================== */

typedef enum {
  JKG_ISA_SCALAR = 0,        /**< No vectorization (portable) */
  JKG_ISA_SSE2 = 1 << 0,     /**< SSE2 (baseline x86_64) */
  JKG_ISA_AVX = 1 << 1,      /**< AVX (256-bit) */
  JKG_ISA_AVX2 = 1 << 2,     /**< AVX2 (256-bit + FMA) */
  JKG_ISA_AVX512F = 1 << 3,  /**< AVX-512 Foundation */
  JKG_ISA_AVX512BW = 1 << 4, /**< AVX-512 Byte/Word */
  JKG_ISA_AMX = 1 << 5,      /**< AMX (tile operations) */
  JKG_ISA_NEON = 1 << 6,     /**< ARM NEON */
  JKG_ISA_SVE = 1 << 7,      /**< ARM SVE */
  JKG_ISA_AUTO = 1 << 31     /**< Automatic ISA selection */
} jkg_isa_t;

/* ========================================================================== */
/* Kernel Types */
/* ========================================================================== */

typedef enum {
  JKG_KERNEL_GEMM_TILE,       /**< GEMM microkernel (MxNxK tile) */
  JKG_KERNEL_GEMM_BIAS,       /**< GEMM + Bias fusion */
  JKG_KERNEL_GEMM_BIAS_RELU,  /**< GEMM + Bias + ReLU fusion */
  JKG_KERNEL_GEMM_BIAS_ACT,   /**< GEMM + Bias + Generic Activation */
  JKG_KERNEL_ELEMENTWISE_ADD, /**< Vectorized element-wise addition */
  JKG_KERNEL_ELEMENTWISE_MUL, /**< Vectorized element-wise multiply */
  JKG_KERNEL_ACTIVATION,      /**< Standalone activation function */
  JKG_KERNEL_REDUCE_SUM,      /**< Reduction: sum across dimension */
  JKG_KERNEL_BATCH_NORM,      /**< Batch normalization */
  JKG_KERNEL_LAYER_NORM,      /**< Layer normalization */
  JKG_KERNEL_CUSTOM           /**< User-defined custom kernel */
} jkg_kernel_type_t;

/* ========================================================================== */
/* Activation Functions */
/* ========================================================================== */

typedef enum {
  JKG_ACT_NONE = 0,
  JKG_ACT_RELU = 1,
  JKG_ACT_RELU6 = 2,
  JKG_ACT_TANH = 3,
  JKG_ACT_SIGMOID = 4,
  JKG_ACT_GELU = 5,
  JKG_ACT_SWISH = 6,
  JKG_ACT_LEAKY_RELU = 7
} jkg_activation_t;

/* ========================================================================== */
/* Vectorization Backend Selection                                            */
/* ========================================================================== */

typedef enum {
  JKG_BACKEND_AUTO = 0,        /**< Automatic selection */
  JKG_BACKEND_HIGHWAY = 1,     /**< Google Highway (portable) */
  JKG_BACKEND_VECTORCLASS = 2, /**< VectorClass (x86 intrinsics) */
  JKG_BACKEND_EVE = 3,         /**< EVE (expressive vector engine) */
  JKG_BACKEND_LLVM = 4         /**< Pure LLVM IR generation */
} jkg_backend_t;

/* ========================================================================== */
/* Kernel Generation Configuration                                            */
/* ========================================================================== */

typedef struct {
  jkg_isa_t target_isa;    /**< Target ISA (or JKG_ISA_AUTO) */
  jkg_backend_t backend;   /**< Vectorization backend */
  int enable_fma;          /**< Enable FMA instructions (1/0) */
  int enable_prefetch;     /**< Enable prefetch hints (1/0) */
  int enable_unroll;       /**< Enable loop unrolling (1/0) */
  int unroll_factor;       /**< Unroll factor (0=auto) */
  size_t cache_line_size;  /**< Cache line size (0=auto-detect) */
  int optimization_level;  /**< LLVM optimization level (0-3) */
  int enable_kernel_cache; /**< Cache compiled kernels (1/0) */
  int verbose;             /**< Verbose logging (1/0) */
} jkg_config_t;

/* ========================================================================== */
/* Kernel Parameters */
/* ========================================================================== */

typedef struct {
  size_t M;                    /**< Matrix/tile dimension M */
  size_t N;                    /**< Matrix/tile dimension N */
  size_t K;                    /**< Matrix/tile dimension K */
  jkg_activation_t activation; /**< Activation function */
  float alpha;                 /**< Scalar multiplier alpha */
  float beta;                  /**< Scalar multiplier beta */
  int has_bias;                /**< Include bias term (1/0) */
  int has_residual;            /**< Include residual connection (1/0) */
  void *custom_params;         /**< Custom parameters (optional) */
} jkg_kernel_params_t;

/* ========================================================================== */
/* Compiled Kernel Handle */
/* ========================================================================== */

struct jkg_kernel_handle_t;

/* ========================================================================== */
/* Kernel Function Signatures */
/* ========================================================================== */

/** GEMM microkernel signature: C = alpha*A*B + beta*C */
typedef void (*jkg_gemm_fn)(const float *A, const float *B, float *C, size_t M,
                            size_t N, size_t K, size_t lda, size_t ldb,
                            size_t ldc, float alpha, float beta);

/** Fused GEMM+Bias kernel signature */
typedef void (*jkg_gemm_bias_fn)(const float *A, const float *B, float *C,
                                 const float *bias, size_t M, size_t N,
                                 size_t K, size_t lda, size_t ldb, size_t ldc,
                                 float alpha);

/** Fused GEMM+Bias+Activation kernel signature */
typedef void (*jkg_gemm_bias_act_fn)(const float *A, const float *B, float *C,
                                     const float *bias, size_t M, size_t N,
                                     size_t K, size_t lda, size_t ldb,
                                     size_t ldc, float alpha);

/** Element-wise operation signature */
typedef void (*jkg_elementwise_fn)(const float *A, const float *B, float *C,
                                   size_t N);

/** Activation function signature */
typedef void (*jkg_activation_fn)(float *data, size_t N);

/* ========================================================================== */
/* Initialization & Configuration */
/* ========================================================================== */

/**
 * @brief Initialize JIT Kernel Generator
 *
 * Initializes LLVM ORC JIT, detects CPU features, and sets up compilation
 * infrastructure. Must be called before any kernel generation.
 *
 * @param config Configuration (NULL = use defaults)
 * @return JKG_OK on success, error code otherwise
 */
int jkg_init(const jkg_config_t *config);

/**
 * @brief Shutdown JIT Kernel Generator
 *
 * Cleanup all resources, invalidate compiled kernels, shutdown LLVM.
 * Safe to call multiple times.
 */
void jkg_shutdown(void);

/**
 * @brief Check if JKG is initialized
 *
 * @return 1 if initialized, 0 otherwise
 */
int jkg_is_initialized(void);

/**
 * @brief Get current configuration
 *
 * @param out_config Pointer to receive current configuration
 * @return JKG_OK on success
 */
int jkg_get_config(jkg_config_t *out_config);

/* ========================================================================== */
/* Kernel Generation */
/* ========================================================================== */

/**
 * @brief Generate and compile a kernel
 *
 * Generates LLVM IR for the specified kernel type, optimizes it, and
 * compiles to native machine code. Returns a handle to the compiled kernel.
 *
 * @param kernel_type Type of kernel to generate
 * @param params Kernel parameters
 * @param out_handle Pointer to receive kernel handle
 * @return JKG_OK on success, error code otherwise
 */
JCORE_API int jkg_generate_kernel(jkg_kernel_type_t kernel_type,
                                  const jkg_kernel_params_t *params,
                                  jkg_kernel_internal_t **out_handle);

/**
 * @brief Get function pointer from compiled kernel
 *
 * Retrieves the native function pointer for a compiled kernel.
 * Cast to appropriate function signature based on kernel type.
 *
 * @param handle Kernel handle
 * @return Function pointer, or NULL on error
 */
JCORE_API void *jkg_get_kernel_function(jkg_kernel_internal_t *handle);

/**
 * @brief Release compiled kernel
 *
 * Frees resources associated with compiled kernel. Function pointer
 * becomes invalid after this call.
 *
 * @param handle Kernel handle to release
 */
JCORE_API void jkg_release_kernel(jkg_kernel_internal_t *handle);

/* ========================================================================== */
/* Convenience Wrappers for Common Patterns */
/* ========================================================================== */

/**
 * @brief Generate GEMM microkernel
 *
 * Creates optimized GEMM tile (MxNxK) with best ISA for current CPU.
 *
 * @param M Tile dimension M (rows)
 * @param N Tile dimension N (columns)
 * @param K Tile dimension K (inner)
 * @param out_handle Pointer to receive kernel handle
 * @return JKG_OK on success
 */
JCORE_API int jkg_generate_gemm_tile(size_t M, size_t N, size_t K,
                                     jkg_kernel_internal_t **out_handle);

/**
 * @brief Generate fused GEMM+Bias+Activation kernel
 *
 * Creates fully fused kernel: C = activation(alpha*A*B + bias)
 *
 * @param M Matrix dimension M
 * @param N Matrix dimension N
 * @param K Matrix dimension K
 * @param activation Activation function
 * @param alpha Scalar multiplier
 * @param out_handle Pointer to receive kernel handle
 * @return JKG_OK on success
 */
JCORE_API int jkg_generate_fused_gemm(size_t M, size_t N, size_t K,
                                      jkg_activation_t activation, float alpha,
                                      jkg_kernel_internal_t **out_handle);

/**
 * @brief Generate element-wise operation kernel
 *
 * Creates vectorized element-wise operation (add/mul/etc.)
 *
 * @param kernel_type JKG_KERNEL_ELEMENTWISE_ADD or _MUL
 * @param N Number of elements
 * @param out_handle Pointer to receive kernel handle
 * @return JKG_OK on success
 */
JCORE_API int jkg_generate_elementwise(jkg_kernel_type_t kernel_type, size_t N,
                                       jkg_kernel_internal_t **out_handle);

/* ========================================================================== */
/* Kernel Cache Management */
/* ========================================================================== */

/**
 * @brief Clear kernel cache
 *
 * Removes all cached compiled kernels, freeing memory.
 * Existing kernel handles remain valid.
 *
 * @return JKG_OK on success
 */
int jkg_clear_cache(void);

/**
 * @brief Get cache statistics
 *
 * @param out_cached Number of cached kernels
 * @param out_hits Cache hit count
 * @param out_misses Cache miss count
 * @return JKG_OK on success
 */
int jkg_get_cache_stats(size_t *out_cached, size_t *out_hits,
                        size_t *out_misses);

/* ========================================================================== */
/* ISA and Feature Query */
/* ========================================================================== */

/**
 * @brief Get detected ISA capabilities
 *
 * Returns bitmask of available ISA features on current CPU.
 *
 * @return ISA bitmask (e.g., JKG_ISA_AVX2 | JKG_ISA_AVX512F)
 */
uint32_t jkg_get_available_isa(void);

/**
 * @brief Get ISA name string
 *
 * @param isa ISA flag
 * @return Human-readable ISA name
 */
const char *jkg_isa_name(jkg_isa_t isa);

/**
 * @brief Get backend name string
 *
 * @param backend Backend type
 * @return Human-readable backend name
 */
const char *jkg_backend_name(jkg_backend_t backend);

/**
 * @brief Get optimal tile sizes for GEMM
 *
 * Determines optimal MxNxK tile sizes based on cache hierarchy
 * and ISA capabilities.
 *
 * @param target_isa Target ISA (or JKG_ISA_AUTO)
 * @param out_M Optimal M dimension
 * @param out_N Optimal N dimension
 * @param out_K Optimal K dimension
 * @return JKG_OK on success
 */
int jkg_get_optimal_tile_sizes(jkg_isa_t target_isa, size_t *out_M,
                               size_t *out_N, size_t *out_K);

/* ========================================================================== */
/* Performance and Debugging */
/* ========================================================================== */

/**
 * @brief Benchmark generated kernel
 *
 * Runs the kernel multiple times and reports performance statistics.
 *
 * @param handle Kernel handle
 * @param iterations Number of iterations
 * @param out_gflops Achieved GFLOPS (for GEMM kernels)
 * @param out_time_ms Average time in milliseconds
 * @return JKG_OK on success
 */
JCORE_API int jkg_benchmark_kernel(jkg_kernel_internal_t *handle,
                                   int iterations, double *out_gflops,
                                   double *out_time_ms);

/**
 * @brief Get LLVM IR for kernel (debugging)
 *
 * Returns the generated LLVM IR as a string. Caller must free with free().
 *
 * @param handle Kernel handle
 * @return LLVM IR string (caller owns), or NULL on error
 */
JCORE_API char *jkg_get_kernel_ir(jkg_kernel_internal_t *handle);

/**
 * @brief Get assembly code for kernel (debugging)
 *
 * Returns the generated native assembly. Caller must free with free().
 *
 * @param handle Kernel handle
 * @return Assembly string (caller owns), or NULL on error
 */
JCORE_API char *jkg_get_kernel_asm(jkg_kernel_internal_t *handle);

/**
 * @brief Dump kernel info to file
 *
 * Writes IR, assembly, and metadata to file for analysis.
 *
 * @param handle Kernel handle
 * @param filename Output file path
 * @return JKG_OK on success
 */
JCORE_API int jkg_dump_kernel(jkg_kernel_internal_t *handle,
                              const char *filename);

/* ========================================================================== */
/* Utility Functions */
/* ========================================================================== */

/**
 * @brief Convert error code to string
 *
 * @param error Error code
 * @return Error message string
 */
const char *jkg_strerror(int error);

/**
 * @brief Get system information
 *
 * Returns string with CPU features, ISA support, cache sizes.
 *
 * @return Static system info string
 */
const char *jkg_get_system_info(void);

/**
 * @brief Run comprehensive self-test
 *
 * Tests kernel generation for all types and ISAs.
 *
 * @param verbose Print detailed results (1/0)
 * @return JKG_OK if all tests pass
 */
int jkg_self_test(int verbose);

#ifdef __cplusplus
}
#endif

#endif /* JCORE_JIT_KERNEL_GENERATOR_H_ */