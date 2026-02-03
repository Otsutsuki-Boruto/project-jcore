// advanced/jcore_neuralPrimitives/src/NPL_core.cpp

#include <cstdio>
#include <cstdlib>
#include "global_thread_scheduler.h"
#include "config.h"
#include "polyhedral_optimization.h"
#include "neural_primitives_internal.h"

namespace npl_internal {

// Global state instance
NPLState g_npl_state;

} // namespace NPL_internal

using namespace npl_internal;

/* ========================================================================== */
/* Initialization Functions                                                    */
/* ========================================================================== */

int npl_init(const npl_config_t *config) {
  std::lock_guard<std::mutex> lock(g_npl_state.state_mutex);

  if (g_npl_state.initialized) {
    return NPL_OK; // Already initialized
  }

  // Set default configuration
  if (config) {
    g_npl_state.config = *config;
  } else {
    npl_config_t default_config = {};
    default_config.num_threads = 0; // Auto-detect
    default_config.enable_fusion = 1;
    default_config.enable_graph_optimization = 1;
    default_config.enable_jit = 1;
    default_config.enable_poly = 1;
    default_config.enable_vectorization = 1;
    default_config.enable_numa = 1;
    default_config.enable_memory_pooling = 1;
    default_config.workspace_size_mb = DEFAULT_WORKSPACE_SIZE_MB;
    default_config.verbose = 0;
    g_npl_state.config = default_config;
  }

  // Detect CPU features and capabilities
  g_npl_state.cpu_info = detect_cpu_info();
  if (g_npl_state.cpu_info.cores == 0) {
    if (g_npl_state.config.verbose) {
      fprintf(stderr, "[NPL] Warning: Failed to detect CPU info\n");
    }
  }

  // Initialize CPU features
  g_npl_state.cpu_features = detect_cpu_features();

  // Get cache information
  g_npl_state.cache_info = ffm_cache_init();
  if (!g_npl_state.cache_info) {
    if (g_npl_state.config.verbose) {
      fprintf(stderr, "[NPL] Warning: Failed to get cache info\n");
    }
  }

  // Determine number of threads
  if (g_npl_state.config.num_threads == 0) {
    g_npl_state.num_threads = g_npl_state.cpu_info.cores;
    if (g_npl_state.num_threads == 0) {
      g_npl_state.num_threads = 1; // Fallback
    }
  } else {
    g_npl_state.num_threads = g_npl_state.config.num_threads;
  }

  // Initialize all dependent components
  int ret = InitializeComponents(&g_npl_state.config);
  if (ret != NPL_OK) {
    fprintf(stderr, "[NPL] Error: Failed to Initialize components: %s\n",
            strerror(ret));
    return ret;
  }

  // Allocate workspace memory
  if (g_npl_state.config.workspace_size_mb > 0) {
    g_npl_state.workspace_size =
        g_npl_state.config.workspace_size_mb * 1024UL * 1024UL;
    g_npl_state.workspace = ffm_aligned_alloc(64, g_npl_state.workspace_size);

    if (!g_npl_state.workspace) {
      fprintf(stderr, "[NPL] Error: Failed to allocate workspace\n");
      ShutdownComponents();
      return NPL_ERR_NO_MEMORY;
    }
  }

  // Precompile common kernels if JIT is enabled
  if (g_npl_state.config.enable_jit) {
    ret = PrecompileCommonKernels();
    if (ret != NPL_OK && g_npl_state.config.verbose) {
      fprintf(stderr, "[NPL] Warning: Failed to precompile kernels\n");
      // Non-fatal - continue without JIT
    }
  }

  g_npl_state.initialized = true;

  if (g_npl_state.config.verbose) {
    printf("[NPL] Initialized successfully\n");
    printf("  Threads: %zu\n", g_npl_state.num_threads);
    printf("  Workspace: %zu MB\n", g_npl_state.config.workspace_size_mb);
    printf("  Fusion: %s\n", g_npl_state.config.enable_fusion ? "ON" : "OFF");
    printf("  JIT: %s\n", g_npl_state.config.enable_jit ? "ON" : "OFF");
    printf("  POLY: %s\n", g_npl_state.config.enable_poly ? "ON" : "OFF");
    printf("  NUMA: %s\n", g_npl_state.config.enable_numa ? "ON" : "OFF");
  }

  return NPL_OK;
}

void npl_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_npl_state.state_mutex);

  if (!g_npl_state.initialized) {
    return; // Not initialized
  }

  if (g_npl_state.config.verbose) {
    printf("[NPL] Shutting down...\n");
    printf("  Total operations: %zu\n",
           g_npl_state.total_ops_executed.load());
    printf("  Operations fused: %zu\n",
           g_npl_state.total_ops_fused.load());
    printf("  Memory saved: %.2f MB\n",
           g_npl_state.total_memory_saved.load() / (1024.0 * 1024.0));
  }

  // Release precompiled kernels
  if (g_npl_state.config.enable_jit) {
    ReleasePrecompiledKernels();
  }

  // Free workspace
  if (g_npl_state.workspace) {
    ffm_free(g_npl_state.workspace);
    g_npl_state.workspace = nullptr;
    g_npl_state.workspace_size = 0;
  }

  // Shutdown all components
  ShutdownComponents();

  // Clear cache info
  if (g_npl_state.cache_info) {
    ffm_cache_free(g_npl_state.cache_info);
    g_npl_state.cache_info = nullptr;
  }

  // Reset state
  g_npl_state.initialized = false;
  g_npl_state.total_ops_executed = 0;
  g_npl_state.total_ops_fused = 0;
  g_npl_state.total_memory_saved = 0;
  g_npl_state.total_gflops = 0.0;
}

int npl_is_initialized(void) {
  return g_npl_state.initialized ? 1 : 0;
}

int npl_get_default_config(npl_config_t *out_config) {
  if (!out_config) {
    return NPL_ERR_INVALID_ARG;
  }

  out_config->num_threads = 0; // Auto-detect
  out_config->enable_fusion = 1;
  out_config->enable_graph_optimization = 1;
  out_config->enable_jit = 1;
  out_config->enable_poly = 1;
  out_config->enable_vectorization = 1;
  out_config->enable_numa = 1;
  out_config->enable_memory_pooling = 1;
  out_config->workspace_size_mb = DEFAULT_WORKSPACE_SIZE_MB;
  out_config->verbose = 0;

  return NPL_OK;
}

/* ========================================================================== */
/* Internal Initialization Functions                                          */
/* ========================================================================== */

namespace npl_internal {

int InitializeComponents(const npl_config_t *config) {

  // Initialize Adaptive Graph Execution Engine
  if (config->enable_graph_optimization) {
    agee_config_t agee_cfg = {};
    agee_cfg.num_threads = config->num_threads;
    agee_cfg.enable_fusion = config->enable_fusion;
    agee_cfg.enable_numa_optimization = config->enable_numa;
    agee_cfg.enable_prefetch = 1;
    agee_cfg.enable_adaptive_tuning = 1;
    agee_cfg.enable_memory_pooling = config->enable_memory_pooling;
    agee_cfg.use_hugepages = 1;
    agee_cfg.memory_pool_size_mb = config->workspace_size_mb / 4;
    agee_cfg.workspace_size_mb = config->workspace_size_mb / 4;
    agee_cfg.fusion_threshold = FUSION_THRESHOLD;
    agee_cfg.verbose = config->verbose;
    agee_cfg.profile_execution = config->verbose;

    if (!agee_is_initialized()) { // Initialize AGEE Only if It is Not Initialized.
      int ret = agee_init(&agee_cfg);
      if (ret != AGEE_OK) {
        fprintf(stderr, "[NPL] Error: Adaptive Execution Engine init failed: %d\n",
                ret);
        return NPL_ERR_INTERNAL;
      }
    }

    // Create execution session
   int ret = agee_create_session(&g_npl_state.agee_session);
    if (ret != AGEE_OK) {
      fprintf(stderr, "[NPL] Error: Failed to create AGEE session: %d\n", ret);
      agee_shutdown();
      return NPL_ERR_INTERNAL;
    }
  }

  // Initialize Polyhedral Optimization Layer
  if (config->enable_poly) {
    poly_opt_config_t poly_config = {};

    poly_config.tile_strategy = POLY_TILE_AUTO;
    poly_config.transform_flags = POLY_TRANSFORM_ALL;

    poly_config.register_tile_M = 4;
    poly_config.register_tile_N = 4;

    poly_config.unroll_factor_outer = 0; // Auto
    poly_config.unroll_factor_inner = 0; // Auto

    poly_config.enable_affine_analysis = 1;
    poly_config.enable_dependency_check = 1;
    poly_config.enable_fusion = 1;
    poly_config.enable_interchange = 1;
    poly_config.enable_vectorization = 1;
    poly_config.enable_prefetch = 1;

    poly_config.cache_occupancy_fraction = 0.75;
    poly_config.cache_line_size = 64;

    poly_config.verbose = 0;
    poly_config.dump_ir = 0;
    poly_config.verify_correctness = 0;

    if (!poly_opt_is_initialized()) { // Initialize Poly Only if It is Not Initialized.
      int ret = poly_opt_init(&poly_config);
      if (ret != POLY_OK) {
        fprintf(stderr, "[NPL] Warning: POLY Initialization Failed failed: %d\n", ret);
        if (g_npl_state.agee_session) {
          agee_destroy_session(g_npl_state.agee_session);
          g_npl_state.agee_session = nullptr;
        }
        if (config->enable_graph_optimization) agee_shutdown();
        return NPL_ERR_INTERNAL;
      }
    }
  }

  // Initialize Global thread scheduler
  try {
    g_npl_state.scheduler = new jcore::global_thread::GlobalThreadScheduler();
    jcore::config::Config thread_cfg;
    thread_cfg.threads = config->num_threads;
    jcore::global_thread::SchedulerResult result = g_npl_state.scheduler->Init(thread_cfg);

    if (!result.ok) {
      fprintf(stderr, "[NPL] Error: Global Thread scheduler init failed: %s\n",
              result.message.c_str());
      delete g_npl_state.scheduler;
      g_npl_state.scheduler = nullptr;
      // Cleanup all initialized components
      if (config->enable_poly) poly_opt_shutdown();
      if (g_npl_state.agee_session) {
        agee_destroy_session(g_npl_state.agee_session);
        g_npl_state.agee_session = nullptr;
      }
      if (config->enable_graph_optimization) {
        agee_shutdown();
      }
      return NPL_ERR_INTERNAL;
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "[NPL] Error: Global Thread scheduler Init failed: %s\n",
            e.what());
    // Cleanup all initialized components
    if (config->enable_poly) poly_opt_shutdown();
    if (g_npl_state.agee_session) {
      agee_destroy_session(g_npl_state.agee_session);
      g_npl_state.agee_session = nullptr;
    }
    if (config->enable_graph_optimization) {
      agee_shutdown();
      poly_opt_shutdown();
    }
    return NPL_ERR_INTERNAL;
  }

  return NPL_OK;
}

void ShutdownComponents() {
  // Shutdown thread scheduler
  if (g_npl_state.scheduler) {
    delete g_npl_state.scheduler;
    g_npl_state.scheduler = nullptr;
  }

  // Shutdown Poly if initialized
  if (g_npl_state.config.enable_poly && poly_opt_is_initialized()){
    poly_opt_shutdown();
  }

  // Shutdown AGEE session and engine
  if (g_npl_state.agee_session) {
    agee_destroy_session(g_npl_state.agee_session);
    g_npl_state.agee_session = nullptr;
  }

  if (g_npl_state.config.enable_graph_optimization && agee_is_initialized()) {
    agee_shutdown();
  }
}

int PrecompileCommonKernels() {
  if (!jkg_is_initialized()) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  g_npl_state.num_jit_kernels = 0;

  // Common GEMM tile sizes
  const size_t gemm_tiles[][3] = {
      {3, 3, 3},     // Small tiles for small matrices
      {8, 8, 8},     // Small tiles for small matrices
      {16, 16, 16},  // Medium tiles
      {32, 32, 32},  // Large tiles for big matrices
      {64, 64, 64},  // Common
      {6, 16, 16},   // Common for transformers
  };

  for (size_t i = 0; i < 4 && g_npl_state.num_jit_kernels < 32; ++i) {
    jkg_kernel_internal_t *handle = nullptr;
    int ret = jkg_generate_gemm_tile(gemm_tiles[i][0], gemm_tiles[i][1],
                                     gemm_tiles[i][2], &handle);
    if (ret == JKG_OK && handle) {
      g_npl_state.jit_kernels[g_npl_state.num_jit_kernels++] = handle;
    }
  }


  // Common fused patterns
  const jkg_activation_t activations[] = {
      JKG_ACT_RELU, JKG_ACT_RELU6, JKG_ACT_GELU, JKG_ACT_SWISH};

  for (size_t i = 0; i < 4 && g_npl_state.num_jit_kernels < 32; ++i) {
    jkg_kernel_internal_t *handle = nullptr;
    int ret = jkg_generate_fused_gemm(16, 16, 16, activations[i], 1.0f, &handle);
    if (ret == JKG_OK && handle) {
      g_npl_state.jit_kernels[g_npl_state.num_jit_kernels++] = handle;
    }
  }

  if (g_npl_state.config.verbose) {
    printf("[NPL] Precompiled %zu JIT kernels\n",
           g_npl_state.num_jit_kernels);
  }

  return NPL_OK;
}

void ReleasePrecompiledKernels() {
  for (size_t i = 0; i < g_npl_state.num_jit_kernels; ++i) {
    if (g_npl_state.jit_kernels[i]) {
      jkg_release_kernel(g_npl_state.jit_kernels[i]);
      g_npl_state.jit_kernels[i] = nullptr;
    }
  }
  g_npl_state.num_jit_kernels = 0;
}

} // namespace NPL_internal

/* ========================================================================== */
/* Utility Functions                                                           */
/* ========================================================================== */

const char *NPL_strerror(int error) {
  switch (error) {
    case NPL_OK: return "Success";
    case NPL_ERR_NOT_INITIALIZED: return "Not initialized";
    case NPL_ERR_INVALID_ARG: return "Invalid argument";
    case NPL_ERR_NO_MEMORY: return "Out of memory";
    case NPL_ERR_INTERNAL: return "Internal error";
    case NPL_ERR_UNSUPPORTED: return "Unsupported operation";
    case NPL_ERR_SHAPE_MISMATCH: return "Tensor shape mismatch";
    case NPL_ERR_OOM: return "Out of memory during execution";
    default: return "Unknown error";
  }
}

const char *npl_get_system_info(void) {
  static char info_buffer[2048];

  if (!g_npl_state.initialized) {
    snprintf(info_buffer, sizeof(info_buffer),
             "JCore Neural Primitives Layer: Not initialized");
    return info_buffer;
  }

  snprintf(info_buffer, sizeof(info_buffer),
           "JCore Neural Primitives Layer\n"
           "  Status: Initialized\n"
           "  Threads: %zu\n"
           "  Workspace: %zu MB\n"
           "  CPU: %d cores (%d logical)\n"
           "  ISA: %s%s%s%s\n"
           "  Features:\n"
           "    Fusion: %s\n"
           "    Graph Optimization: %s\n"
           "    JIT Compilation: %s (%zu kernels)\n"
           "    Polyhedral Optimization: %s\n"
           "    NUMA: %s\n"
           "  Statistics:\n"
           "    Total Operations: %zu\n"
           "    Fused Operations: %zu\n"
           "    Memory Saved: %.2f MB\n"
           "  Backends:\n"
           "    MIL: %s\n"
           "    KFE: %s\n"
           "    AGEE: %s\n",
           g_npl_state.num_threads,
           g_npl_state.workspace_size / (1024 * 1024),
           g_npl_state.cpu_info.cores,
           g_npl_state.cpu_info.logical_cores,
           g_npl_state.cpu_features.avx2 ? " AVX2" : "",
           g_npl_state.cpu_features.avx512 ? " AVX512" : "",
           g_npl_state.cpu_features.avx ? " AVX" : "",
           g_npl_state.cpu_features.amx ? " AMX" : "",
           g_npl_state.config.enable_fusion ? "ON" : "OFF",
           g_npl_state.config.enable_graph_optimization ? "ON" : "OFF",
           g_npl_state.config.enable_jit ? "ON" : "OFF",
           g_npl_state.num_jit_kernels,
           g_npl_state.config.enable_poly ? "ON" : "OFF",
           g_npl_state.config.enable_numa ? "ON" : "OFF",
           g_npl_state.total_ops_executed.load(),
           g_npl_state.total_ops_fused.load(),
           g_npl_state.total_memory_saved.load() / (1024.0 * 1024.0),
           mil_is_initialized() ? mil_backend_name(mil_get_backend()) : "OFF",
           kfe_is_initialized() ? "ON" : "OFF",
           agee_is_initialized() ? "ON" : "OFF");

  return info_buffer;
}