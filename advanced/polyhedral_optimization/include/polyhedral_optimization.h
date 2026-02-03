// advanced/polyhedral_optimization/include/polyhedral_optimization.h
#ifndef JCORE_POLYHEDRAL_OPTIMIZATION_H_
#define JCORE_POLYHEDRAL_OPTIMIZATION_H_

/**
 * @file polyhedral_optimization.h
 * @brief Polyhedral Optimization Layer - Automatic Loop Tiling/Vectorization
 *
 * Component: Polyhedral Optimization Layer (Advanced)
 * Purpose: Applies automatic loop tiling/vectorization
 *
 * Dependencies:
 *   Advanced:
 *     - JIT Kernel Generator: IR generation and compilation infrastructure
 *   Base:
 *     - Cache Blocking and Tiling Utility: Compute L1/L2/L3-optimal block sizes
 *
 * Features:
 *   - Automatic loop tiling based on cache hierarchy
 *   - Polyhedral model-based transformations (affine scheduling)
 *   - Loop interchange, fusion, and distribution
 *   - Vectorization-friendly loop transformations
 *   - Data locality optimization
 *   - Register tiling and software pipelining
 *   - Multi-level tiling (register, L1, L2, L3)
 *
 * Requirements:
 *   - LLVM 20.x
 *   - C++17 compiler
 *   - JIT Kernel Generator (LLVM-20 based)
 *   - Cache Blocking Utility (FFM API)
 *
 * Thread-safety: Thread-safe after initialization
 * FFM API: Fully compatible with Project JCore FFM API
 */

#include <cstddef>
#include <cstdint>

/* Export macro for shared library */
#ifdef _WIN32
#ifdef JCORE_BUILD_DLL
#define JCORE_POLY_API __declspec(dllexport)
#else
#define JCORE_POLY_API __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define JCORE_POLY_API __attribute__((visibility("default")))
#else
#define JCORE_POLY_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration of opaque handle */
struct poly_opt_context_t;
struct poly_opt_plan_t;

/* ========================================================================== */
/* Error Codes */
/* ========================================================================== */

#define POLY_OK 0
#define POLY_ERR_NOT_INITIALIZED -1
#define POLY_ERR_INVALID_ARG -2
#define POLY_ERR_NO_MEMORY -3
#define POLY_ERR_INTERNAL -4
#define POLY_ERR_LLVM_ERROR -5
#define POLY_ERR_UNSUPPORTED_LOOP -6
#define POLY_ERR_DEPENDENCY_VIOLATION -7
#define POLY_ERR_CACHE_INFO_FAILED -8
#define POLY_ERR_JIT_FAILED -9
#define POLY_ERR_INVALID_IR -10
#define POLY_ERR_NO_AFFINE_LOOPS -11

/* ========================================================================== */
/* Tiling Strategy */
/* ========================================================================== */

typedef enum {
  POLY_TILE_NONE = 0,        /**< No tiling */
  POLY_TILE_REGISTER = 1,    /**< Register-level tiling (innermost) */
  POLY_TILE_L1 = 2,          /**< L1 cache tiling */
  POLY_TILE_L2 = 3,          /**< L2 cache tiling */
  POLY_TILE_L3 = 4,          /**< L3 cache tiling */
  POLY_TILE_MULTI_LEVEL = 5, /**< Multi-level hierarchical tiling */
  POLY_TILE_AUTO = 6         /**< Automatic tile size selection */
} poly_tile_strategy_t;

/* ========================================================================== */
/* Loop Transformation Types */
/* ========================================================================== */

typedef enum {
  POLY_TRANSFORM_NONE = 0,
  POLY_TRANSFORM_INTERCHANGE = 1 << 0,  /**< Loop interchange */
  POLY_TRANSFORM_FUSION = 1 << 1,       /**< Loop fusion */
  POLY_TRANSFORM_DISTRIBUTION = 1 << 2, /**< Loop distribution */
  POLY_TRANSFORM_UNROLL = 1 << 3,       /**< Loop unrolling */
  POLY_TRANSFORM_JAM = 1 << 4,          /**< Loop unroll-and-jam */
  POLY_TRANSFORM_PEEL = 1 << 5,         /**< Loop peeling */
  POLY_TRANSFORM_VECTORIZE = 1 << 6,    /**< Enable vectorization */
  POLY_TRANSFORM_PREFETCH = 1 << 7,     /**< Insert prefetch hints */
  POLY_TRANSFORM_ALL = 0xFFFF           /**< Enable all transformations */
} poly_transform_flags_t;

/* ========================================================================== */
/* Optimization Configuration */
/* ========================================================================== */

typedef struct {
  poly_tile_strategy_t tile_strategy; /**< Tiling strategy */
  uint32_t transform_flags;           /**< Transformation flags (bitmask) */

  /* Tile sizes (0 = auto-compute) */
  size_t tile_size_M; /**< M dimension tile size */
  size_t tile_size_N; /**< N dimension tile size */
  size_t tile_size_K; /**< K dimension tile size */

  /* Register tiling */
  size_t register_tile_M; /**< Register-level M tile (e.g., 4-8) */
  size_t register_tile_N; /**< Register-level N tile (e.g., 4-8) */

  /* Unrolling factors */
  int unroll_factor_outer; /**< Outer loop unroll factor (0=auto) */
  int unroll_factor_inner; /**< Inner loop unroll factor (0=auto) */

  /* Optimization levels */
  int enable_affine_analysis;  /**< Enable polyhedral affine analysis */
  int enable_dependency_check; /**< Enable loop dependency checking */
  int enable_fusion;           /**< Enable loop fusion */
  int enable_interchange;      /**< Enable loop interchange */
  int enable_vectorization;    /**< Enable vectorization hints */
  int enable_prefetch;         /**< Enable prefetch insertion */

  /* Cache configuration */
  double cache_occupancy_fraction; /**< Fraction of cache to use (0.0-1.0) */
  size_t cache_line_size;          /**< Cache line size (0=auto-detect) */

  /* Debugging */
  int verbose;            /**< Verbose logging */
  int dump_ir;            /**< Dump IR before/after optimization */
  int verify_correctness; /**< Run correctness verification */
} poly_opt_config_t;

/* ========================================================================== */
/* Loop Nest Information */
/* ========================================================================== */

typedef struct {
  size_t loop_depth;         /**< Number of nested loops */
  size_t *trip_counts;       /**< Trip count for each loop level */
  size_t *strides;           /**< Memory access strides */
  int is_perfectly_nested;   /**< Perfect loop nest? */
  int has_affine_bounds;     /**< All bounds affine? */
  int has_data_dependencies; /**< Has loop-carried dependencies? */
  void *llvm_loop_info;      /**< LLVM LoopInfo (internal) */
} poly_loop_info_t;

/* ========================================================================== */
/* Optimization Statistics */
/* ========================================================================== */

typedef struct {
  size_t loops_analyzed;          /**< Number of loops analyzed */
  size_t loops_tiled;             /**< Number of loops tiled */
  size_t loops_vectorized;        /**< Number of loops vectorized */
  size_t loops_interchanged;      /**< Number of loops interchanged */
  size_t loops_fused;             /**< Number of loops fused */
  double optimization_time_ms;    /**< Time spent optimizing (ms) */
  double expected_speedup;        /**< Expected speedup factor */
  size_t memory_accesses_reduced; /**< Estimated memory access reduction */
} poly_opt_stats_t;

/* ========================================================================== */
/* Initialization & Configuration */
/* ========================================================================== */

/**
 * @brief Initialize polyhedral optimization layer
 *
 * Initializes LLVM Polly infrastructure, cache analysis, and JIT integration.
 * Must be called before any optimization operations.
 *
 * @param config Configuration (NULL = use defaults)
 * @return POLY_OK on success, error code otherwise
 */
JCORE_POLY_API int poly_opt_init(const poly_opt_config_t *config);


JCORE_POLY_API int poly_opt_run_polly_passes(poly_opt_plan_t *plan);
/**
 * @brief Shutdown polyhedral optimization layer
 *
 * Cleanup all resources, invalidate optimization plans.
 */
JCORE_POLY_API void poly_opt_shutdown(void);

/**
 * @brief Check if polyhedral optimizer is initialized
 *
 * @return 1 if initialized, 0 otherwise
 */
JCORE_POLY_API int poly_opt_is_initialized(void);

/**
 * @brief Get current configuration
 *
 * @param out_config Pointer to receive current configuration
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_get_config(poly_opt_config_t *out_config);

/**
 * @brief Update configuration (can be called after init)
 *
 * @param config New configuration
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_set_config(const poly_opt_config_t *config);

/* ========================================================================== */
/* Optimization Plan Creation */
/* ========================================================================== */

/**
 * @brief Create optimization plan for LLVM function
 *
 * Analyzes LLVM IR function and creates optimization plan based on
 * loop structure, memory access patterns, and cache hierarchy.
 *
 * @param llvm_function Pointer to LLVM Function (void* cast from
 * llvm::Function*)
 * @param out_plan Pointer to receive optimization plan handle
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_create_plan(void *llvm_function,
                                        poly_opt_plan_t **out_plan);

/**
 * @brief Create optimization plan for JIT kernel
 *
 * Convenience wrapper for JIT-generated kernels.
 *
 * @param jit_kernel_handle Handle from JIT Kernel Generator
 * @param out_plan Pointer to receive optimization plan handle
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_create_plan_from_jit(void *jit_kernel_handle,
                                                 poly_opt_plan_t **out_plan);

/**
 * @brief Release optimization plan
 *
 * @param plan Plan handle to release
 */
JCORE_POLY_API void poly_opt_release_plan(poly_opt_plan_t *plan);

/* ========================================================================== */
/* Loop Analysis */
/* ========================================================================== */

/**
 * @brief Analyze loop nest structure
 *
 * Performs polyhedral analysis on loop nest to determine:
 * - Affine loop bounds
 * - Memory access patterns
 * - Data dependencies
 * - Optimization opportunities
 *
 * @param plan Optimization plan
 * @param out_info Pointer to receive loop information
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_analyze_loops(poly_opt_plan_t *plan,
                                          poly_loop_info_t *out_info);

/**
 * @brief Check if loop nest is tileable
 *
 * Verifies that loop nest has affine bounds and no dependencies
 * that would prevent tiling.
 *
 * @param plan Optimization plan
 * @return 1 if tileable, 0 otherwise
 */
JCORE_POLY_API int poly_opt_is_tileable(poly_opt_plan_t *plan);

/**
 * @brief Get recommended tile sizes for loop nest
 *
 * Computes optimal tile sizes based on cache hierarchy and
 * memory access patterns.
 *
 * @param plan Optimization plan
 * @param level Cache level (POLY_TILE_L1, L2, or L3)
 * @param out_tile_M M dimension tile size
 * @param out_tile_N N dimension tile size
 * @param out_tile_K K dimension tile size
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_recommend_tile_sizes(poly_opt_plan_t *plan,
                                                 poly_tile_strategy_t level,
                                                 size_t *out_tile_M,
                                                 size_t *out_tile_N,
                                                 size_t *out_tile_K);

/* ========================================================================== */
/* Loop Transformations */
/* ========================================================================== */

/**
 * @brief Apply loop tiling transformation
 *
 * Applies cache-aware loop tiling to the loop nest in the plan.
 * Modifies the LLVM IR in-place.
 *
 * @param plan Optimization plan
 * @param strategy Tiling strategy
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_apply_tiling(poly_opt_plan_t *plan,
                                         poly_tile_strategy_t strategy);

/**
 * @brief Apply loop interchange
 *
 * Reorders loops to improve spatial locality.
 *
 * @param plan Optimization plan
 * @param loop_order Array of loop indices (outer to inner)
 * @param num_loops Number of loops to reorder
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_apply_interchange(poly_opt_plan_t *plan,
                                              const size_t *loop_order,
                                              size_t num_loops);

/**
 * @brief Apply loop unrolling
 *
 * Unrolls loops by specified factors.
 *
 * @param plan Optimization plan
 * @param outer_factor Outer loop unroll factor
 * @param inner_factor Inner loop unroll factor
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_apply_unrolling(poly_opt_plan_t *plan,
                                            int outer_factor, int inner_factor);

/**
 * @brief Apply vectorization transformation
 *
 * Prepares loops for vectorization and inserts vectorization hints.
 *
 * @param plan Optimization plan
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_apply_vectorization(poly_opt_plan_t *plan);

/**
 * @brief Apply all enabled transformations
 *
 * Applies all transformations specified in configuration in optimal order.
 *
 * @param plan Optimization plan
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_apply_all_transforms(poly_opt_plan_t *plan);

/* ========================================================================== */
/* LLVM Integration */
/* ========================================================================== */

/**
 * @brief Get optimized LLVM function
 *
 * Returns pointer to the optimized LLVM Function.
 *
 * @param plan Optimization plan
 * @return Pointer to llvm::Function (as void*), or NULL on error
 */
JCORE_POLY_API void *poly_opt_get_optimized_function(poly_opt_plan_t *plan);

/**
 * @brief Get optimized LLVM IR as string
 *
 * Returns the optimized IR for debugging. Caller must free with free().
 *
 * @param plan Optimization plan
 * @return IR string (caller owns), or NULL on error
 */
JCORE_POLY_API char *poly_opt_get_ir_string(poly_opt_plan_t *plan);

/* ========================================================================== */
/* Performance Analysis */
/* ========================================================================== */

/**
 * @brief Get optimization statistics
 *
 * Returns statistics about the optimizations performed.
 *
 * @param plan Optimization plan
 * @param out_stats Pointer to receive statistics
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_get_stats(poly_opt_plan_t *plan,
                                      poly_opt_stats_t *out_stats);

/**
 * @brief Estimate performance improvement
 *
 * Estimates the expected speedup from optimizations based on
 * cache model and roofline analysis.
 *
 * @param plan Optimization plan
 * @return Expected speedup factor (1.0 = no improvement)
 */
JCORE_POLY_API double poly_opt_estimate_speedup(poly_opt_plan_t *plan);

/* ========================================================================== */
/* Correctness Verification */
/* ========================================================================== */

/**
 * @brief Verify correctness of optimized code
 *
 * Runs both original and optimized versions with test data and
 * compares results for correctness.
 *
 * @param plan Optimization plan
 * @param test_iterations Number of test iterations
 * @return 1 if correct, 0 if mismatch detected
 */
JCORE_POLY_API int poly_opt_verify_correctness(poly_opt_plan_t *plan,
                                               int test_iterations);

/* ========================================================================== */
/* Utility Functions */
/* ========================================================================== */

/**
 * @brief Convert error code to string
 *
 * @param error Error code
 * @return Error message string
 */
JCORE_POLY_API const char *poly_opt_strerror(int error);

/**
 * @brief Get cache information summary
 *
 * Returns string with detected cache sizes and optimal tile sizes.
 *
 * @return Static cache info string
 */
JCORE_POLY_API const char *poly_opt_get_cache_info(void);

/**
 * @brief Run self-test
 *
 * Tests polyhedral optimization on synthetic kernels.
 *
 * @param verbose Print detailed results (1/0)
 * @return POLY_OK if all tests pass
 */
JCORE_POLY_API int poly_opt_self_test(int verbose);

/* ========================================================================== */
/* High-Level Convenience Functions */
/* ========================================================================== */

/**
 * @brief Optimize GEMM loop nest
 *
 * High-level function to optimize a GEMM-style triple nested loop.
 * Applies multi-level tiling, interchange, and vectorization.
 *
 * @param llvm_function Pointer to LLVM Function containing GEMM loops
 * @param M Matrix dimension M
 * @param N Matrix dimension N
 * @param K Matrix dimension K
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_optimize_gemm(void *llvm_function, size_t M,
                                          size_t N, size_t K);

/**
 * @brief Optimize convolution loop nest
 *
 * High-level function to optimize convolution loops with im2col.
 *
 * @param llvm_function Pointer to LLVM Function containing conv loops
 * @param channels Number of channels
 * @param height Image height
 * @param width Image width
 * @param kernel_size Kernel size
 * @return POLY_OK on success
 */
JCORE_POLY_API int poly_opt_optimize_conv(void *llvm_function, size_t channels,
                                          size_t height, size_t width,
                                          size_t kernel_size);

#ifdef __cplusplus
}
#endif

#endif /* JCORE_POLYHEDRAL_OPTIMIZATION_H_ */