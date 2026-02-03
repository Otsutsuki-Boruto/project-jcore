// advanced/polyhedral_optimization/src/polyhedral_core.cpp

/**
 * @file polyhedral_core.cpp
 * @brief Core polyhedral optimization infrastructure and state management
 *
 * Handles initialization, configuration, LLVM infrastructure setup,
 * and cache information management for polyhedral transformations.
 */

#include "polyhedral_optimization.h"
#include "polyhedral_internal.h"

#include <llvm/Analysis/ScalarEvolution.h>

#include <cstdio>
#include <cstdarg>
#include <cstring>

namespace poly_internal {

// Global state instance
PolyOptState g_poly_state;

/* ========================================================================== */
/* Logging Utilities */
/* ========================================================================== */

void log_error(const char *format, ...) {
  if (!g_poly_state.config.verbose) return;

  fprintf(stderr, "[POLY ERROR] ");
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fprintf(stderr, "\n");
  fflush(stderr);
}

void log_info(const char *format, ...) {
  if (!g_poly_state.config.verbose) return;

  fprintf(stdout, "[POLY INFO] ");
  va_list args;
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);
  fprintf(stdout, "\n");
  fflush(stdout);
}

void log_debug(const char *format, ...) {
  if (!g_poly_state.config.verbose) return;

  fprintf(stdout, "[POLY DEBUG] ");
  va_list args;
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);
  fprintf(stdout, "\n");
  fflush(stdout);
}

/* ========================================================================== */
/* Cache Information Management */
/* ========================================================================== */

bool initialize_cache_info() {
  log_info("Initializing cache information");

  // Initialize FFM cache info
  g_poly_state.ffm_cache_info = ffm_cache_init();
  if (!g_poly_state.ffm_cache_info) {
    log_error("Failed to initialize FFM cache info");
    return false;
  }

  if (g_poly_state.config.verbose) {
    ffm_cache_print(g_poly_state.ffm_cache_info);
  }

  // Compute default tile sizes for each cache level
  size_t elem_size = sizeof(float); // Default to float
  double occupancy = g_poly_state.config.cache_occupancy_fraction;

  // L1 tile
  g_poly_state.default_l1_tile = ffm_cache_compute_tile(
      g_poly_state.ffm_cache_info, 1, elem_size, occupancy);

  // L2 tile
  g_poly_state.default_l2_tile = ffm_cache_compute_tile(
      g_poly_state.ffm_cache_info, 2, elem_size, occupancy);

  // L3 tile
  g_poly_state.default_l3_tile = ffm_cache_compute_tile(
      g_poly_state.ffm_cache_info, 3, elem_size, occupancy);

  log_info("Default tile sizes: L1=%zu, L2=%zu, L3=%zu",
           g_poly_state.default_l1_tile,
           g_poly_state.default_l2_tile,
           g_poly_state.default_l3_tile);

  return true;
}

void cleanup_cache_info() {
  if (g_poly_state.ffm_cache_info) {
    ffm_cache_free(g_poly_state.ffm_cache_info);
    g_poly_state.ffm_cache_info = nullptr;
  }
}

/* ========================================================================== */
/* Configuration Management */
/* ========================================================================== */

poly_opt_config_t get_default_config() {
  poly_opt_config_t config = {};

  config.tile_strategy = POLY_TILE_AUTO;
  config.transform_flags = POLY_TRANSFORM_ALL;

  // Tile sizes (0 = auto-compute)
  config.tile_size_M = 0;
  config.tile_size_N = 0;
  config.tile_size_K = 0;

  // Register tiling
  config.register_tile_M = 4;
  config.register_tile_N = 4;

  // Unrolling
  config.unroll_factor_outer = 0; // Auto
  config.unroll_factor_inner = 0; // Auto

  // Optimization flags
  config.enable_affine_analysis = 1;
  config.enable_dependency_check = 1;
  config.enable_fusion = 1;
  config.enable_interchange = 1;
  config.enable_vectorization = 1;
  config.enable_prefetch = 1;

  // Cache configuration
  config.cache_occupancy_fraction = 0.75;
  config.cache_line_size = 64; // Will be auto-detected

  // Debugging
  config.verbose = 0;
  config.dump_ir = 0;
  config.verify_correctness = 0;

  return config;
}

/* ========================================================================== */
/* Error Handling */
/* ========================================================================== */

const char *error_to_string(int error) {
  switch (error) {
    case POLY_OK: return "Success";
    case POLY_ERR_NOT_INITIALIZED: return "Polyhedral optimizer not initialized";
    case POLY_ERR_INVALID_ARG: return "Invalid argument";
    case POLY_ERR_NO_MEMORY: return "Out of memory";
    case POLY_ERR_INTERNAL: return "Internal error";
    case POLY_ERR_LLVM_ERROR: return "LLVM error";
    case POLY_ERR_UNSUPPORTED_LOOP: return "Unsupported loop structure";
    case POLY_ERR_DEPENDENCY_VIOLATION: return "Loop dependency violation";
    case POLY_ERR_CACHE_INFO_FAILED: return "Failed to get cache info";
    case POLY_ERR_JIT_FAILED: return "JIT compilation failed";
    case POLY_ERR_INVALID_IR: return "Invalid LLVM IR";
    case POLY_ERR_NO_AFFINE_LOOPS: return "No affine loops found";
    default: return "Unknown error";
  }
}

} // namespace poly_internal

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

extern "C" {

int poly_opt_init(const poly_opt_config_t *config) {
  using namespace poly_internal;

  std::lock_guard<std::mutex> lock(g_poly_state.state_mutex);

  if (g_poly_state.initialized) {
    log_info("Polyhedral optimizer already initialized");
    return POLY_OK;
  }

  log_info("Initializing polyhedral optimization layer");

  // Set configuration
  if (config) {
    g_poly_state.config = *config;
  } else {
    g_poly_state.config = get_default_config();
  }

  // Initialize cache information
  if (!initialize_cache_info()) {
    log_error("Failed to initialize cache information");
    return POLY_ERR_CACHE_INFO_FAILED;
  }

  // Initialize JIT Kernel Generator (if not already initialized)
  if (!jkg_is_initialized()) {
    log_info("Initializing JIT Kernel Generator");
    jkg_config_t jit_config = {};
    jit_config.target_isa = JKG_ISA_AUTO;
    jit_config.backend = JKG_BACKEND_AUTO;
    jit_config.enable_fma = 1;
    jit_config.enable_prefetch = 1;
    jit_config.enable_unroll = 1;
    jit_config.unroll_factor = 0; // AUTO
    jit_config.cache_line_size = 64;
    jit_config.optimization_level = 2;
    jit_config.enable_kernel_cache = 1;
    jit_config.verbose = g_poly_state.config.verbose;

    int ret = jkg_init(&jit_config);
    if (ret != JKG_OK) {
      log_error("Failed to initialize JIT Kernel Generator: %d", ret);
      cleanup_cache_info();
      return POLY_ERR_JIT_FAILED;
    }
    printf("[POLY INFO] JIT Kernel Generator Initialized\n");
  }

  // Reset statistics
  g_poly_state.stats.total_plans_created = 0;
  g_poly_state.stats.total_loops_optimized = 0;
  g_poly_state.stats.total_optimization_time_ms = 0.0;

  g_poly_state.initialized = true;
  log_info("Polyhedral optimization layer initialized successfully");

  return POLY_OK;
}

void poly_opt_shutdown(void) {
  using namespace poly_internal;

  std::lock_guard<std::mutex> lock(g_poly_state.state_mutex);

  if (!g_poly_state.initialized) {
    return;
  }

  log_info("Shutting down polyhedral optimization layer");

  // Clear all optimization plans
  {
    std::lock_guard<std::mutex> plan_lock(g_poly_state.plan_mutex);
    g_poly_state.active_plans.clear();
  }

  // Cleanup cache info
  cleanup_cache_info();

  g_poly_state.initialized = false;
  log_info("Polyhedral optimization layer shut down");
}

int poly_opt_is_initialized(void) {
  return poly_internal::g_poly_state.initialized ? 1 : 0;
}

int poly_opt_get_config(poly_opt_config_t *out_config) {
  using namespace poly_internal;

  if (!out_config) {
    return POLY_ERR_INVALID_ARG;
  }

  if (!g_poly_state.initialized) {
    return POLY_ERR_NOT_INITIALIZED;
  }

  std::lock_guard<std::mutex> lock(g_poly_state.state_mutex);
  *out_config = g_poly_state.config;

  return POLY_OK;
}

int poly_opt_set_config(const poly_opt_config_t *config) {
  using namespace poly_internal;

  if (!config) {
    return POLY_ERR_INVALID_ARG;
  }

  if (!g_poly_state.initialized) {
    return POLY_ERR_NOT_INITIALIZED;
  }

  std::lock_guard<std::mutex> lock(g_poly_state.state_mutex);

  // Validate configuration
  if (config->cache_occupancy_fraction <= 0.0 ||
      config->cache_occupancy_fraction > 1.0) {
    log_error("Invalid cache occupancy fraction: %f",
              config->cache_occupancy_fraction);
    return POLY_ERR_INVALID_ARG;
  }

  g_poly_state.config = *config;
  log_info("Configuration updated");

  return POLY_OK;
}

const char *poly_opt_strerror(int error) {
  return poly_internal::error_to_string(error);
}

const char *poly_opt_get_cache_info(void) {
  using namespace poly_internal;

  static char info_buffer[512];

  if (!g_poly_state.initialized || !g_poly_state.ffm_cache_info) {
    snprintf(info_buffer, sizeof(info_buffer),
             "Cache info not initialized");
    return info_buffer;
  }

  snprintf(info_buffer, sizeof(info_buffer),
           "Cache Tile Sizes: L1=%zu, L2=%zu, L3=%zu (elements)",
           g_poly_state.default_l1_tile,
           g_poly_state.default_l2_tile,
           g_poly_state.default_l3_tile);

  return info_buffer;
}

int poly_opt_self_test(int verbose) {
  using namespace poly_internal;

  int original_verbose = g_poly_state.config.verbose;
  if (verbose) {
    g_poly_state.config.verbose = 1;
  }

  log_info("Running polyhedral optimization self-test");

  // Test 1: Initialization
  if (!g_poly_state.initialized) {
    poly_opt_config_t config = get_default_config();
    config.verbose = verbose;
    int ret = poly_opt_init(&config);
    if (ret != POLY_OK) {
      log_error("Self-test failed: initialization");
      return ret;
    }
  }

  // Test 2: Cache info
  if (!g_poly_state.ffm_cache_info) {
    log_error("Self-test failed: cache info not initialized");
    return POLY_ERR_CACHE_INFO_FAILED;
  }

  // Test 3: Default tile sizes
  if (g_poly_state.default_l1_tile == 0 ||
      g_poly_state.default_l2_tile == 0 ||
      g_poly_state.default_l3_tile == 0) {
    log_error("Self-test failed: invalid default tile sizes");
    return POLY_ERR_INTERNAL;
  }

  log_info("Self-test passed: L1=%zu, L2=%zu, L3=%zu",
           g_poly_state.default_l1_tile,
           g_poly_state.default_l2_tile,
           g_poly_state.default_l3_tile);

  g_poly_state.config.verbose = original_verbose;

  return POLY_OK;
}

} // extern "C"