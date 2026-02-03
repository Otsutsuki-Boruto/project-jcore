// advanced/src/agee_core.cpp
#include "agee_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mem_wrapper.h"
#include "profiler_api.h"
#include "vmath_engine.h"
#include "global_thread_scheduler.h"
#include "config.h"

namespace agee_internal {

// Global state definition
AGEEGlobalState g_agee_state;

/* ========================================================================== */
/* Internal Component Initialization                                          */
/* ========================================================================== */

int InitializeComponents(const agee_config_t *config) {
  std::lock_guard<std::mutex> lock(g_agee_state.global_mutex);


  // Initialize Operator Graph Runtime (handles KFE and dispatch internally)
  bool og_was_initialized = og_is_initialized();
  if (!og_was_initialized) {
    og_config_t og_cfg = {};
    og_cfg.enable_fusion = config->enable_fusion;
    og_cfg.enable_parallelism = 1;
    og_cfg.enable_memory_reuse = 1;
    og_cfg.enable_pattern_matching = 1;
    og_cfg.max_fusion_depth = MAX_FUSION_DEPTH;
    og_cfg.num_threads = config->num_threads;
    og_cfg.verbose = config->verbose;
    og_cfg.fusion_threshold = config->fusion_threshold;

    if (og_init(&og_cfg) != OG_OK) {
      fprintf(stderr, "[AGEE] Failed to initialize Operator Graph Runtime\n");
      return AGEE_ERR_INTERNAL;
    }

    // Mark all OG-managed components as initialized
    g_agee_state.og_initialized = true;
    g_agee_state.kfe_initialized = true;
    g_agee_state.dispatch_initialized = true;
  }

  // Initialize Vector Math Engine (independent component)
  if (vmath_init() != VMATH_OK) {
    fprintf(stderr, "[AGEE] Failed to initialize Vector Math Engine\n");
    og_shutdown();  // ← Cleanup OG before returning
    g_agee_state.og_initialized = false;
    g_agee_state.kfe_initialized = false;
    g_agee_state.dispatch_initialized = false;
    return AGEE_ERR_INTERNAL;
  }
  g_agee_state.vmath_initialized = true;

  // Initialize Profiler (independent component)
  if (!jcore::profiler::Init()) {
    if (config->verbose) {
      fprintf(stderr, "[AGEE] Warning: Profiler initialization failed\n");
    }
    g_agee_state.profiler_initialized = false;
  } else {
    g_agee_state.profiler_initialized = true;
  }

  // Initialize cache info
  g_agee_state.cache_info = ffm_cache_init();
  if (!g_agee_state.cache_info && config->verbose) {
    fprintf(stderr, "[AGEE] Warning: Cache info initialization failed\n");
  }

  // Detect CPU info
  g_agee_state.cpu_info = detect_cpu_info();

  if (config->verbose) {
    printf("[AGEE] Initialization complete:\n");
    printf("  OG Runtime: initialized (includes KFE + Dispatch)\n");
    printf("  Cores: %d\n", g_agee_state.cpu_info.cores);
    printf("  NUMA nodes: %d\n", g_agee_state.cpu_info.numa_nodes);
    printf("  AVX2: %s\n", g_agee_state.cpu_info.avx2 ? "Yes" : "No");
    printf("  AVX-512: %s\n", g_agee_state.cpu_info.avx512 ? "Yes" : "No");
    printf("  L3 cache: %d KB\n", g_agee_state.cpu_info.l3_kb);
  }

  return AGEE_OK;
}

  void ShutdownComponents() {
  std::lock_guard<std::mutex> lock(g_agee_state.global_mutex);

  // Shutdown independent components first (in reverse initialization order)
  if (g_agee_state.profiler_initialized) {
    jcore::profiler::Shutdown();
    g_agee_state.profiler_initialized = false;
  }

  if (g_agee_state.vmath_initialized) {
    vmath_shutdown();
    g_agee_state.vmath_initialized = false;
  }

  // Shutdown OG (which handles KFE and dispatch internally)
  if (g_agee_state.og_initialized) {
    og_shutdown();  // This calls kfe_shutdown() and k_dispatch_shutdown()

    // Reset all OG-managed component flags
    g_agee_state.og_initialized = false;
    g_agee_state.kfe_initialized = false;      // ← KFE was shutdown by og_shutdown
    g_agee_state.dispatch_initialized = false; // ← Dispatch was shutdown by og_shutdown
  }

  // Cleanup AGEE-owned resources
  if (g_agee_state.cache_info) {
    ffm_cache_free(g_agee_state.cache_info);
    g_agee_state.cache_info = nullptr;
  }
}

} // namespace agee_internal

using namespace agee_internal;

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

extern "C" {

int agee_init(const agee_config_t *config) {
  if (g_agee_state.initialized) {
    return AGEE_OK; // Already initialized
  }

  // Use default config if none provided
  agee_config_t default_cfg;
  if (!config) {
    agee_get_default_config(&default_cfg);
    config = &default_cfg;
  }

  // Copy configuration
  memcpy(&g_agee_state.global_config, config, sizeof(agee_config_t));

  // Initialize NUMA manager if enabled
  if (config->enable_numa_optimization) {
    if (numa_manager_init() != 0 && config->verbose) {
      fprintf(stderr, "[AGEE] Warning: NUMA manager initialization failed\n");
    }
  }

  // Initialize all components
  int ret = InitializeComponents(config);
  if (ret != AGEE_OK) {
    return ret;
  }

  g_agee_state.initialized = true;

  if (config->verbose) {
    printf("[AGEE] Adaptive Graph Execution Engine initialized successfully\n");
  }

  return AGEE_OK;
}

void agee_shutdown(void) {
  if (!g_agee_state.initialized) {
    return;
  }

  // Destroy all active sessions first
  {
    std::lock_guard<std::mutex> lock(g_agee_state.global_mutex);

    // Create a copy of sessions vector to avoid iterator invalidation
    std::vector<SessionImpl *> sessions_copy = g_agee_state.sessions;
    g_agee_state.sessions.clear();

    // Destroy each session
    for (auto *session : sessions_copy) {
      if (session) {
        // Manually cleanup session without using agee_destroy_session
        // to avoid mutex deadlock

        // Destroy plans
        for (auto *plan : session->active_plans) {
          if (plan) {
            FreePlanMemory(session, plan);
            delete plan;
          }
        }
        session->active_plans.clear();

        // Cleanup components
        if (session->thread_scheduler) {
          session->thread_scheduler->Shutdown();
          delete session->thread_scheduler;
        }
        if (session->cached_tuner) {
          cat_shutdown(session->cached_tuner);
        }
        if (session->memory_pool) {
          pm_shutdown(session->memory_pool);
        }

        delete session;
      }
    }
  }

  // Shutdown components
  ShutdownComponents();

  // Shutdown NUMA manager
  if (g_agee_state.global_config.enable_numa_optimization) {
    numa_manager_shutdown();
  }

  g_agee_state.initialized = false;

  if (g_agee_state.global_config.verbose) {
    printf("[AGEE] Shutdown complete\n");
  }
}

int agee_is_initialized(void) { return g_agee_state.initialized ? 1 : 0; }

int agee_get_default_config(agee_config_t *out_config) {
  if (!out_config) {
    return AGEE_ERR_INVALID_ARG;
  }

  memset(out_config, 0, sizeof(agee_config_t));

  // Default configuration
  out_config->num_threads = 0; // Auto-detect
  out_config->enable_fusion = 1;
  out_config->enable_numa_optimization = 1;
  out_config->enable_prefetch = 1;
  out_config->enable_adaptive_tuning = 1;
  out_config->enable_memory_pooling = 1;
  out_config->use_hugepages = 1;
  out_config->memory_pool_size_mb = DEFAULT_POOL_SIZE_MB;
  out_config->workspace_size_mb = DEFAULT_WORKSPACE_SIZE_MB;
  out_config->fusion_threshold = DEFAULT_FUSION_THRESHOLD;
  out_config->verbose = 0;
  out_config->profile_execution = 1;

  return AGEE_OK;
}

int agee_create_session(agee_session_t *out_session) {
  if (!out_session) {
    return AGEE_ERR_INVALID_ARG;
  }

  if (!g_agee_state.initialized) {
    return AGEE_ERR_NOT_INITIALIZED;
  }

  // Allocate session
  SessionImpl *session = new (std::nothrow) SessionImpl();
  if (!session) {
    return AGEE_ERR_NO_MEMORY;
  }

  // Generate session ID
  session->session_id = g_agee_state.next_session_id.fetch_add(1);

  // Copy global config
  memcpy(&session->config, &g_agee_state.global_config, sizeof(agee_config_t));
  memcpy(&session->cpu_info, &g_agee_state.cpu_info, sizeof(cpu_info_t));
  session->max_numa_nodes = g_agee_state.cpu_info.numa_nodes;

  // Initialize memory pool if enabled
  if (session->config.enable_memory_pooling) {
    size_t pool_size = session->config.memory_pool_size_mb * 1024 * 1024;
    size_t chunk_size = 64 * 1024; // 64KB chunks
    int use_hp = session->config.use_hugepages ? 1 : 0;

    pm_status_t st =
        pm_init(&session->memory_pool, pool_size, chunk_size, use_hp, -1);
    if (st != PM_OK && session->config.verbose) {
      fprintf(stderr, "[AGEE] Warning: Memory pool initialization failed\n");
    }
  }

  // Initialize cached auto-tuner if enabled
  if (session->config.enable_adaptive_tuning) {
    cat_status_t st = cat_init(&session->cached_tuner);
    if (st != CAT_OK && session->config.verbose) {
      fprintf(stderr, "[AGEE] Warning: Cached tuner Initialization failed\n");
    }
  }

  // Initialize thread scheduler
  session->thread_scheduler =
      new (std::nothrow) jcore::global_thread::GlobalThreadScheduler();
  if (session->thread_scheduler) {
    jcore::config::Config thread_cfg;
    thread_cfg.threads = session->config.num_threads;
    jcore::global_thread::SchedulerResult result = session->thread_scheduler->Init(thread_cfg);
    if (!result.ok && session->config.verbose) {
      fprintf(stderr, "[AGEE] Warning: Global Thread scheduler Init: %s\n",
              result.message.c_str());
    }
  }

  // Register session
  {
    std::lock_guard<std::mutex> lock(g_agee_state.global_mutex);
    g_agee_state.sessions.push_back(session);
  }

  *out_session = reinterpret_cast<agee_session_t>(session);

  if (session->config.verbose) {
    printf("[AGEE] Session %lu created\n", session->session_id);
  }

  return AGEE_OK;
}

  void agee_destroy_session(agee_session_t session) {
  if (!session) {
    return;
  }

  SessionImpl *impl = reinterpret_cast<SessionImpl *>(session);

  // Enforce strict lifecycle: session must not own active plans
  {
    std::lock_guard<std::mutex> lock(impl->session_mutex);
    if (!impl->active_plans.empty()) {
      fprintf(stderr,
              "[AGEE ERROR] Destroying session with %zu active plans\n",
              impl->active_plans.size());
      std::abort();  // contract violation (debug + HPC correctness)
    }
  }

  // Cleanup thread scheduler
  if (impl->thread_scheduler) {
    impl->thread_scheduler->Shutdown();
    delete impl->thread_scheduler;
    impl->thread_scheduler = nullptr;
  }

  // Cleanup cached tuner
  if (impl->cached_tuner) {
    cat_shutdown(impl->cached_tuner);
    impl->cached_tuner = nullptr;
  }

  // Cleanup memory pool
  if (impl->memory_pool) {
    pm_shutdown(impl->memory_pool);
    impl->memory_pool = nullptr;
  }

  // Unregister session
  {
    std::lock_guard<std::mutex> lock(g_agee_state.global_mutex);
    auto it = std::find(g_agee_state.sessions.begin(),
                        g_agee_state.sessions.end(), impl);
    if (it != g_agee_state.sessions.end()) {
      g_agee_state.sessions.erase(it);
    }
  }

  if (impl->config.verbose) {
    printf("[AGEE] Session %lu destroyed\n", impl->session_id);
  }

  delete impl;
}

int agee_reset_session(agee_session_t session) {
  if (!session) {
    return AGEE_ERR_INVALID_ARG;
  }

  SessionImpl *impl = reinterpret_cast<SessionImpl *>(session);
  std::lock_guard<std::mutex> lock(impl->session_mutex);

  // Clear cached tuner results if available
  if (impl->cached_tuner) {
    cat_clear_cache(impl->cached_tuner);
  }

  // Reset statistics
  memset(&impl->cumulative_stats, 0, sizeof(agee_exec_stats_t));
  impl->total_executions.store(0);

  return AGEE_OK;
}

const char *agee_strerror(int error) {
  switch (error) {
  case AGEE_OK:
    return "Success";
  case AGEE_ERR_NOT_INITIALIZED:
    return "AGEE not initialized";
  case AGEE_ERR_INVALID_ARG:
    return "Invalid argument";
  case AGEE_ERR_NO_MEMORY:
    return "Out of memory";
  case AGEE_ERR_INTERNAL:
    return "Internal error";
  case AGEE_ERR_GRAPH_INVALID:
    return "Invalid graph";
  case AGEE_ERR_EXECUTION_FAILED:
    return "Execution failed";
  case AGEE_ERR_OOM:
    return "Out of memory during execution";
  case AGEE_ERR_UNSUPPORTED:
    return "Unsupported operation";
  default:
    return "Unknown error";
  }
}

void agee_free_string(char *str) {
  if (str) {
    free(str);
  }
}

} // extern "C"