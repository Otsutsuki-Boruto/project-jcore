// advanced/kFusion_engine/src/kernel_fusion_engine_core.cpp

#include "kernel_fusion_engine_internal.h"
#include <new>

#include "adaptive_tuner.h"
#include "jcore_isa_dispatch.h"
#include "mem_wrapper.h"
#include "global_thread_scheduler.h"
#include "config.h"

namespace kfe_internal
{
  // Define global state
  KFEState g_kfe_state;
}

using namespace kfe_internal;

extern "C"
{

  /* ========================================================================== */
  /* Initialization & Shutdown                                                  */
  /* ========================================================================== */

  int kfe_init(const kfe_config_t *config)
  {
    std::lock_guard<std::mutex> lock(g_kfe_state.state_mutex);

    if (g_kfe_state.initialized)
    {
      return KFE_OK; // Already initialized
    }

    // Set default config
    if (config)
    {
      g_kfe_state.config = *config;
    }
    else
    {
      g_kfe_state.config.num_threads = 0; // auto
      g_kfe_state.config.enable_vectorization = 1;
      g_kfe_state.config.enable_cache_blocking = 1;
      g_kfe_state.config.enable_prefetch = 1;
      g_kfe_state.config.enable_kernel_autotuning = 1;
      g_kfe_state.config.workspace_size_mb = 256; // 256 MB default
      g_kfe_state.config.verbose = 0;
    }

    // Initialize global thread scheduler
    g_kfe_state.scheduler = new (std::nothrow) jcore::global_thread::GlobalThreadScheduler();
    if (!g_kfe_state.scheduler)
    {
      pm_shutdown(g_kfe_state.pool_manager);
      ffm_cache_free(g_kfe_state.cache_info);
      mil_shutdown();
      return KFE_ERR_NO_MEMORY;
    }

    jcore::config::Config cfg;
    cfg.threads = (g_kfe_state.config.num_threads == 0) ? 0 : g_kfe_state.config.num_threads;
    jcore::global_thread::SchedulerResult result = g_kfe_state.scheduler->Init(cfg);
    if (!result.ok)
    {
      delete g_kfe_state.scheduler;
      pm_shutdown(g_kfe_state.pool_manager);
      ffm_cache_free(g_kfe_state.cache_info);
      mil_shutdown();
      return KFE_ERR_INTERNAL;
    }

    // Initialize microkernel interface
    mil_config_t mil_cfg = {};
    mil_cfg.preferred_backend = MIL_BACKEND_AUTO;
    mil_cfg.num_threads = g_kfe_state.config.num_threads;
    mil_cfg.enable_prefetch = g_kfe_state.config.enable_prefetch;
    mil_cfg.enable_auto_tuning = g_kfe_state.config.enable_kernel_autotuning;
    mil_cfg.verbose = g_kfe_state.config.verbose;

    if (mil_init(&mil_cfg) != MIL_OK)
    {
      ffm_shutdown();
      return KFE_ERR_INTERNAL;
    }

    // Initialize memory pool manager
    size_t pool_bytes = g_kfe_state.config.workspace_size_mb * 1024 * 1024;
    size_t chunk_bytes = 4096; // 4 KB chunks
    pm_status_t pm_st = pm_init(&g_kfe_state.pool_manager, pool_bytes, chunk_bytes, 0, -1);
    if (pm_st != PM_OK)
    {
      ffm_cache_free(g_kfe_state.cache_info);
      mil_shutdown();
      return KFE_ERR_INTERNAL;
    }

    g_kfe_state.initialized = true;

    return KFE_OK;
  }

  void kfe_shutdown(void)
  {
    std::lock_guard<std::mutex> lock(g_kfe_state.state_mutex);

    if (!g_kfe_state.initialized)
    {
      return;
    }

    if (g_kfe_state.scheduler)
    {
      g_kfe_state.scheduler->Shutdown();
      delete g_kfe_state.scheduler;
      g_kfe_state.scheduler = nullptr;
    }

    if (g_kfe_state.pool_manager)
    {
      pm_shutdown(g_kfe_state.pool_manager);
      g_kfe_state.pool_manager = nullptr;
    }

    mil_shutdown();

    g_kfe_state.initialized = false;
  }

  int kfe_is_initialized(void)
  {
    return g_kfe_state.initialized ? 1 : 0;
  }

  int kfe_set_num_threads(size_t num_threads)
  {
    if (!g_kfe_state.initialized)
    {
      return KFE_ERR_NOT_INITIALIZED;
    }

    mil_set_num_threads(num_threads);
    g_kfe_state.config.num_threads = num_threads;

    return KFE_OK;
  }

  size_t kfe_get_num_threads(void)
  {
    if (!g_kfe_state.initialized || !g_kfe_state.scheduler)
    {
      return 1;
    }
    return g_kfe_state.scheduler->GetNumThreads();
  }

} // extern "C"