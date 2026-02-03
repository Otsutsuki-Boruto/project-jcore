// advanced/polyhedral_optimization/include/polyhedral_internal.h
#ifndef POLYHEDRAL_INTERNAL_H_
#define POLYHEDRAL_INTERNAL_H_

/**
 * @file polyhedral_internal.h
 * @brief Internal structures and utilities for polyhedral optimization
 */

#include "polyhedral_optimization.h"

// JIT Kernel Generator headers
#include "jit_kernel_internal.h"

// Base component headers
#include "ffm_cache_block.h"

// LLVM headers
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>

// Standard C++ headers
#include <cstring>
#include <vector>
#include <memory>
#include <mutex>
#include <string>

namespace poly_internal {
  struct PolyOptPlanImpl;

/* ========================================================================== */
/* Forward Declarations */
/* ========================================================================== */

struct PolyOptState;
struct LoopNestInfo;
struct TilingStrategy;

/* ========================================================================== */
/* Loop Nest Information (Internal) */
/* ========================================================================== */

struct LoopNestInfo {
  llvm::Loop *root_loop;                    // Root of loop nest
  std::vector<llvm::Loop*> nested_loops;    // All loops in nest (outer to inner)
  std::vector<size_t> trip_counts;          // Trip counts for each loop
  std::vector<size_t> strides;              // Memory access strides
  std::vector<llvm::PHINode*> induction_vars; // Induction variables

  bool is_perfectly_nested;
  bool has_affine_bounds;
  bool has_data_dependencies;
  bool is_tileable;

  // Memory access analysis
  size_t num_loads;
  size_t num_stores;
  std::vector<llvm::Value*> memory_accesses;

  LoopNestInfo()
      : root_loop(nullptr),
        is_perfectly_nested(false),
        has_affine_bounds(false),
        has_data_dependencies(false),
        is_tileable(false),
        num_loads(0),
        num_stores(0) {}
};

/* ========================================================================== */
/* Tiling Configuration */
/* ========================================================================== */

struct TilingConfig {
  poly_tile_strategy_t strategy;

  // Tile sizes for each dimension
  size_t tile_M;
  size_t tile_N;
  size_t tile_K;

  // Register-level tile sizes
  size_t reg_tile_M;
  size_t reg_tile_N;

  // Multi-level tiling
  bool enable_multi_level;
  size_t l1_tile_M, l1_tile_N, l1_tile_K;
  size_t l2_tile_M, l2_tile_N, l2_tile_K;
  size_t l3_tile_M, l3_tile_N, l3_tile_K;

  TilingConfig()
      : strategy(POLY_TILE_AUTO),
        tile_M(0), tile_N(0), tile_K(0),
        reg_tile_M(4), reg_tile_N(4),
        enable_multi_level(false),
        l1_tile_M(0), l1_tile_N(0), l1_tile_K(0),
        l2_tile_M(0), l2_tile_N(0), l2_tile_K(0),
        l3_tile_M(0), l3_tile_N(0), l3_tile_K(0) {}
};

/* ========================================================================== */
/* Optimization Plan Implementation */
/* ========================================================================== */

struct PolyOptPlanImpl {
  llvm::Function *function;           // LLVM function being optimized
  llvm::Module *module;               // Containing module
  std::unique_ptr<llvm::LoopInfo> loop_info;
  std::unique_ptr<llvm::ScalarEvolution> scalar_evolution;

  LoopNestInfo nest_info;             // Loop nest analysis results
  TilingConfig tiling_config;         // Tiling configuration
  poly_opt_stats_t stats;             // Optimization statistics

  // Transformation state
  bool analysis_done;
  bool tiling_applied;
  bool interchange_applied;
  bool vectorization_applied;
  bool unrolling_applied;

  // Original function (for verification)
  llvm::Function *original_function;

  // Timing
  double analysis_time_ms;
  double transform_time_ms;

  PolyOptPlanImpl()
      : function(nullptr),
        module(nullptr),
        analysis_done(false),
        tiling_applied(false),
        interchange_applied(false),
        vectorization_applied(false),
        unrolling_applied(false),
        original_function(nullptr),
        analysis_time_ms(0.0),
        transform_time_ms(0.0) {
    std::memset(&stats, 0, sizeof(stats));
  }
};

/* ========================================================================== */
/* Global State */
/* ========================================================================== */

struct GlobalStats {
  size_t total_plans_created;
  size_t total_loops_optimized;
  double total_optimization_time_ms;

  GlobalStats()
      : total_plans_created(0),
        total_loops_optimized(0),
        total_optimization_time_ms(0.0) {}
};

struct PolyOptState {
  bool initialized;
  poly_opt_config_t config;

  // Cache information
  ffm_cache_info_t *ffm_cache_info;
  size_t default_l1_tile;
  size_t default_l2_tile;
  size_t default_l3_tile;

  // Active optimization plans
  std::vector<std::shared_ptr<PolyOptPlanImpl>> active_plans;
  std::mutex plan_mutex;

  // Global statistics
  GlobalStats stats;
  std::mutex state_mutex;

  PolyOptState()
      : initialized(false),
        ffm_cache_info(nullptr),
        default_l1_tile(0),
        default_l2_tile(0),
        default_l3_tile(0) {}
};

// Global state instance (defined in polyhedral_core.cpp)
extern PolyOptState g_poly_state;

/* ========================================================================== */
/* Utility Functions */
/* ========================================================================== */

// LLVM Initialization
void initialize_llvm_targets();

// Logging
void log_error(const char *format, ...);
void log_info(const char *format, ...);
void log_debug(const char *format, ...);

// Cache management
bool initialize_cache_info();
void cleanup_cache_info();
poly_opt_config_t get_default_config();

// Error handling
const char *error_to_string(int error);

// Casting helpers for opaque pointer conversion
inline PolyOptPlanImpl *plan_to_impl(poly_opt_plan_t *plan) {
  return reinterpret_cast<PolyOptPlanImpl*>(plan);
}

inline poly_opt_plan_t *impl_to_plan(PolyOptPlanImpl *impl) {
  return reinterpret_cast<poly_opt_plan_t*>(impl);
}

/* ========================================================================== */
/* Loop Analysis Functions (polyhedral_tiling.cpp) */
/* ========================================================================== */

// Analyze loop nest structure and memory access patterns
int analyze_loop_nest(PolyOptPlanImpl *plan);

// Check if loop nest is tileable
bool check_tileability(PolyOptPlanImpl *plan);

// Compute optimal tile sizes based on cache hierarchy
void compute_tile_sizes(PolyOptPlanImpl *plan, poly_tile_strategy_t strategy);

// Extract loop bounds and trip counts
bool extract_loop_bounds(llvm::Loop *loop, llvm::ScalarEvolution *SE,
                         size_t *trip_count);

/* ========================================================================== */
/* Tiling Transformation Functions (polyhedral_tiling.cpp) */
/* ========================================================================== */

// Apply loop tiling transformation
int apply_tiling_transform(PolyOptPlanImpl *plan);

// Apply multi-level tiling
int apply_multilevel_tiling(PolyOptPlanImpl *plan);

// Create tiled loop structure
llvm::Loop *create_tiled_loop(llvm::Loop *original_loop,
                              size_t tile_size,
                              llvm::IRBuilder<> *builder);

/* ========================================================================== */
/* Loop Interchange Functions (polyhedral_llvm_integration.cpp) */
/* ========================================================================== */

// Apply loop interchange transformation
int apply_interchange_transform(PolyOptPlanImpl *plan,
                                const std::vector<size_t> &loop_order);

// Check if interchange is legal
bool is_interchange_legal(const std::vector<llvm::Loop*> &loops,
                         const std::vector<size_t> &new_order);

/* ========================================================================== */
/* Vectorization Functions (polyhedral_vectorization.cpp) */
/* ========================================================================== */

// Prepare loops for vectorization
int prepare_for_vectorization(PolyOptPlanImpl *plan);

// Insert vectorization hints
void insert_vectorization_hints(llvm::Loop *loop);

// Check if loop is vectorizable
bool is_vectorizable(llvm::Loop *loop);

// Internal use only: applies prefetching transformations
int apply_prefetching(PolyOptPlanImpl *plan);

/* ========================================================================== */
/* Unrolling Functions (polyhedral_vectorization.cpp) */
/* ========================================================================== */

// Apply loop unrolling
int apply_unrolling_transform(PolyOptPlanImpl *plan,
                               int outer_factor,
                               int inner_factor);

// Compute optimal unroll factor
int compute_unroll_factor(llvm::Loop *loop, size_t trip_count);

/* ========================================================================== */
/* LLVM IR Utilities (polyhedral_llvm_integration.cpp) */
/* ========================================================================== */

// Clone function for transformation
llvm::Function *clone_function(llvm::Function *original);

// Get loop information
llvm::LoopInfo *compute_loop_info(llvm::Function *func);

// Get scalar evolution analysis
llvm::ScalarEvolution *compute_scalar_evolution(llvm::Function *func);

// Dump IR to string
std::string get_ir_string(llvm::Function *func);

// Verify IR validity
bool verify_function(llvm::Function *func);

/* ========================================================================== */
/* Performance Estimation */
/* ========================================================================== */

// Estimate speedup from optimizations
double estimate_speedup(const PolyOptPlanImpl *plan);

// Estimate memory access reduction
size_t estimate_memory_reduction(const LoopNestInfo &info,
                                 const TilingConfig &config);

} // namespace poly_internal

#endif /* POLYHEDRAL_INTERNAL_H_ */