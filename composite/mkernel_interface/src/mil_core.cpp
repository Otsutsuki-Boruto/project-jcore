#include "microkernel_interface.h"
#include "k_kernel_dispatch.h"
#include "vmath_engine.h"
#include "cpu_info.h"
#include "jcore_isa_dispatch.h"
#include "ffm_cache_block.h"
#include "mem_wrapper.h"
#include "pool_manager.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <mutex>
#include <string>

#include "cached_autotuner.h"

// Include OpenMP header at file scope to avoid nested extern "C" issues
#ifdef _OPENMP
#include <omp.h>
#endif

/* ========================================================================== */
/* Internal State                                                              */
/* ========================================================================== */

namespace
{

  cat_handle_t *g_trcs_handle = nullptr;

  struct MILState
  {
    bool initialized;
    mil_backend_t active_backend;
    size_t num_threads;
    bool enable_prefetch;
    bool enable_auto_tuning;
    bool verbose;
    cpu_info_t cpu_info;
    ffm_cache_info_t *cache_info;
    std::mutex mutex;
    pm_t *pool_manager;

    MILState()
        : initialized(false),
          active_backend(MIL_BACKEND_AUTO),
          num_threads(0),
          enable_prefetch(true),
          enable_auto_tuning(true),
          verbose(false),
          cpu_info{},
          cache_info(nullptr),
          pool_manager(nullptr)
    {
    }

    ~MILState()
    {
      if (cache_info)
      {
        ffm_cache_free(cache_info);
        cache_info = nullptr;
      }
      if (pool_manager)
      {
        pm_shutdown(pool_manager);
        pool_manager = nullptr;
      }
    }
  };

  MILState g_mil_state;

} // anonymous namespace

/* ========================================================================== */
/* Backend Detection - Use Kernel Dispatch as Source of Truth                 */
/* ========================================================================== */

/**
 * @brief Query kernel dispatch to determine which backends are available
 *
 * Instead of re-detecting backends, we trust the kernel dispatch layer
 * which has already done proper detection and registration.
 */
static mil_backend_t detect_best_backend_from_dispatch()
{
  // Perform a test dispatch to trigger kernel selection
  float test_a[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float test_b[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float test_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] Performing test dispatch to detect backend...\n");
  }

  // This will trigger kernel selection in dispatch layer
  int status = k_dispatch_matmul(test_a, test_b, test_c, 2, 2, 2);

  if (status != JCORE_OK)
  {
    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] Test dispatch failed with status %d\n", status);
    }
    return MIL_BACKEND_FALLBACK;
  }

  // Now query what kernel was selected
  const char *selected_kernel = k_dispatch_get_last_selected_kernel();

  if (selected_kernel == nullptr)
  {
    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] No kernel name returned from dispatch\n");
    }
    return MIL_BACKEND_FALLBACK;
  }

  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] Dispatch selected kernel: %s\n", selected_kernel);
  }

  // Map kernel name to backend
  if (std::strstr(selected_kernel, "openblas") != nullptr)
  {
    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] Mapping to OpenBLAS backend\n");
    }
    return MIL_BACKEND_OPENBLAS;
  }
  else if (std::strstr(selected_kernel, "blis") != nullptr)
  {
    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] Mapping to BLIS backend\n");
    }
    return MIL_BACKEND_BLIS;
  }

  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] Unknown kernel name, using fallback\n");
  }

  return MIL_BACKEND_FALLBACK;
}

/* ========================================================================== */
/* Backend Initialization                                                      */
/* ========================================================================== */

static int init_openblas()
{
  extern int openblas_get_num_threads() __attribute__((weak));
  extern void openblas_set_num_threads(int) __attribute__((weak));

  if (openblas_set_num_threads == nullptr)
  {
    return MIL_ERR_NO_BACKEND;
  }

  int num_threads = static_cast<int>(g_mil_state.num_threads);
  if (num_threads == 0)
  {
    num_threads = g_mil_state.cpu_info.logical_cores;
  }

  openblas_set_num_threads(num_threads);

  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] OpenBLAS Initialized with %d threads\n", num_threads);
  }

  return MIL_OK;
}

static int init_blis()
{
// BLIS typically uses OpenMP for threading (already configured)
#ifdef _OPENMP
  int num_threads = static_cast<int>(g_mil_state.num_threads);
  if (num_threads == 0)
  {
    num_threads = g_mil_state.cpu_info.logical_cores;
  }
  omp_set_num_threads(num_threads);

  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] BLIS Initialized with %d threads (via OpenMP)\n", num_threads);
  }
#else
  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] BLIS backend ready (OpenMP not available)\n");
  }
#endif
  return MIL_OK;
}

/* ========================================================================== */
/* Public API: Initialization                                                  */
/* ========================================================================== */

int mil_init(const mil_config_t *config)
{
  std::lock_guard<std::mutex> lock(g_mil_state.mutex);

  if (g_mil_state.initialized)
  {
    return MIL_OK; // Already initialized
  }

  // Apply configuration
  if (config != nullptr)
  {
    g_mil_state.active_backend = config->preferred_backend;
    g_mil_state.num_threads = config->num_threads;
    g_mil_state.enable_prefetch = config->enable_prefetch;
    g_mil_state.enable_auto_tuning = config->enable_auto_tuning;
    g_mil_state.verbose = config->verbose;
  }
  else
  {
    // Use defaults
    g_mil_state.active_backend = MIL_BACKEND_AUTO;
    g_mil_state.num_threads = 0; // Auto-detect
    g_mil_state.enable_prefetch = true;
    g_mil_state.enable_auto_tuning = true;
    g_mil_state.verbose = false;
  }

  // Initialize memory allocator wrapper
  int status = ffm_init(FFM_BACKEND_AUTO);
  if (status != FFM_OK)
  {
    std::fprintf(stderr, "[MIL] Failed to Initialize memory allocator\n");
    return MIL_ERR_INTERNAL;
  }

  // Initialize memory pool manager
  size_t pool_size = 128 * 1024 * 1024; // 128 MB
  size_t chunk_size = 64 * 1024; // 64 KB chunks
  pm_status_t pm_status = pm_init(&g_mil_state.pool_manager, pool_size, chunk_size, 0, -1);
  if (pm_status != PM_OK)
  {
    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] Warning: Pool manager init failed, using FFM allocator\n");
    }
    g_mil_state.pool_manager = nullptr;
  }

  // Detect CPU features
  g_mil_state.cpu_info = detect_cpu_info();

  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] CPU Info: %d cores, %d logical cores\n",
                 g_mil_state.cpu_info.cores, g_mil_state.cpu_info.logical_cores);
    std::fprintf(stderr, "[MIL] Features: AVX=%d AVX2=%d AVX512=%d AMX=%d\n",
                 g_mil_state.cpu_info.avx, g_mil_state.cpu_info.avx2,
                 g_mil_state.cpu_info.avx512, g_mil_state.cpu_info.amx);
    std::fprintf(stderr, "[MIL] Cache: L1D=%dKB L2=%dKB L3=%dKB\n",
                 g_mil_state.cpu_info.l1d_kb, g_mil_state.cpu_info.l2_kb,
                 g_mil_state.cpu_info.l3_kb);
  }

  // Initialize cache info for tile computation
  g_mil_state.cache_info = ffm_cache_init();
  if (g_mil_state.cache_info == nullptr)
  {
    std::fprintf(stderr, "[MIL] Warning: Failed to Initialize cache info\n");
  }

  // Initialize base components
  status = jcore_init_dispatch();
  if (status != JCORE_OK)
  {
    std::fprintf(stderr, "[MIL] Failed to Initialize ISA dispatch\n");
    return MIL_ERR_INTERNAL;
  }

  // Initialize derived components
  status = k_dispatch_init();
  if (status != JCORE_OK)
  {
    std::fprintf(stderr, "[MIL] Failed to Initialize kernel dispatch\n");
    return MIL_ERR_INTERNAL;
  }

  status = vmath_init();
  if (status != VMATH_OK)
  {
    std::fprintf(stderr, "[MIL] Failed to Initialize vector math\n");
    return MIL_ERR_INTERNAL;
  }

  /* ===================== TRCS (Cached AutoTuner) Init + Load ===================== */

  if (g_mil_state.enable_auto_tuning)
  {
    cat_config_t cat_cfg;
    cat_config_init_default(&cat_cfg);

    std::snprintf(cat_cfg.cache_dir, sizeof(cat_cfg.cache_dir),
                  "%s", "./.mil_trcs_cache");
    cat_cfg.format = CAT_FORMAT_BINARY;   // or CAT_FORMAT_JSON
    cat_cfg.max_entries = 0;              // unlimited
    cat_cfg.validate_hardware = 1;
    cat_cfg.auto_save = 1;
    cat_cfg.force_benchmark = 0;

    cat_status_t st = cat_init_with_config(&g_trcs_handle, &cat_cfg);
    if (st != CAT_OK)
    {
      std::fprintf(stderr, "[MIL] Warning: TRCS init failed (%s)\n",
                   cat_status_str(st));
    }
    else
    {
      st = cat_load_cache(g_trcs_handle);
      if (st != CAT_OK && g_mil_state.verbose)
      {
        std::fprintf(stderr, "[MIL] TRCS cache load failed (%s)\n",
                     cat_status_str(st));
      }
      else if (g_mil_state.verbose)
      {
        std::fprintf(stderr, "[MIL] TRCS cache loaded successfully\n");
      }
    }
  }

  /* ============================================================================== */

  // Auto-detect backend if needed - USE KERNEL DISPATCH AS SOURCE OF TRUTH
  if (g_mil_state.active_backend == MIL_BACKEND_AUTO)
  {
    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] Querying kernel dispatch for available backends...\n");
    }
    g_mil_state.active_backend = detect_best_backend_from_dispatch();
    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] Selected backend: %s\n",
                   mil_backend_name(g_mil_state.active_backend));
    }
  }

  // Initialize selected backend
  int backend_status = MIL_ERR_NO_BACKEND;
  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] Initializing %s backend:\n",
                 mil_backend_name(g_mil_state.active_backend));
  }

  switch (g_mil_state.active_backend)
  {
  case MIL_BACKEND_OPENBLAS:
    backend_status = init_openblas();
    if (backend_status != MIL_OK && g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] OpenBLAS initialization failed (status=%d)\n", backend_status);
    }
    break;
  case MIL_BACKEND_BLIS:
    backend_status = init_blis();
    if (backend_status != MIL_OK && g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] BLIS initialization failed (status=%d)\n", backend_status);
    }
    break;
  case MIL_BACKEND_FALLBACK:
    backend_status = MIL_OK; // Fallback always available
    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] WARNING: Using fallback - expect low performance!\n");
    }
    break;
  default:
    break;
  }

  if (backend_status != MIL_OK)
  {
    std::fprintf(stderr, "[MIL] ERROR: Backend Initialization failed, falling back to portable implementation\n");
    std::fprintf(stderr, "[MIL] This will result in VERY LOW performance (10-50x slower than optimized BLAS)\n");
    g_mil_state.active_backend = MIL_BACKEND_FALLBACK;
  }

  g_mil_state.initialized = true;

  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] Initialization complete. Backend: %s\n",
                 mil_backend_name(g_mil_state.active_backend));
  }

  return MIL_OK;
}

void mil_shutdown()
{
  std::lock_guard<std::mutex> lock(g_mil_state.mutex);

  if (!g_mil_state.initialized)
  {
    return;
  }

  // Cleanup cache info
  if (g_mil_state.cache_info != nullptr)
  {
    ffm_cache_free(g_mil_state.cache_info);
    g_mil_state.cache_info = nullptr;
  }

  // Shutdown pool manager
  if (g_mil_state.pool_manager != nullptr)
  {
    pm_shutdown(g_mil_state.pool_manager);
    g_mil_state.pool_manager = nullptr;
  }

  // Shutdown derived components

  /* ===================== TRCS (Cached AutoTuner) Shutdown ===================== */
  if (g_trcs_handle != nullptr)
  {
    cat_status_t st = cat_save_cache(g_trcs_handle);
    if (st != CAT_OK && g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] TRCS cache save failed (%s)\n",
                   cat_status_str(st));
    }

    cat_shutdown(g_trcs_handle);
    g_trcs_handle = nullptr;

    if (g_mil_state.verbose)
    {
      std::fprintf(stderr, "[MIL] TRCS shutdown complete\n");
    }
  }
  /* ============================================================================ */

  k_dispatch_shutdown();
  vmath_shutdown();

  // Shutdown memory allocator
  ffm_shutdown();

  g_mil_state.initialized = false;

  if (g_mil_state.verbose)
  {
    std::fprintf(stderr, "[MIL] Shutdown complete\n");
  }
}

int mil_is_initialized()
{
  return g_mil_state.initialized ? 1 : 0;
}

mil_backend_t mil_get_backend()
{
  return g_mil_state.active_backend;
}

const char *mil_backend_name(mil_backend_t backend)
{
  switch (backend)
  {
  case MIL_BACKEND_AUTO:
    return "Auto";
  case MIL_BACKEND_OPENBLAS:
    return "OpenBLAS";
  case MIL_BACKEND_BLIS:
    return "BLIS";
  case MIL_BACKEND_FALLBACK:
    return "Fallback";
  default:
    return "Unknown";
  }
}

int mil_set_num_threads(size_t num_threads)
{
  if (!g_mil_state.initialized)
  {
    return MIL_ERR_NOT_INITIALIZED;
  }

  g_mil_state.num_threads = num_threads;

  // Update backend thread count
  switch (g_mil_state.active_backend)
  {
  case MIL_BACKEND_OPENBLAS:
  {
    extern void openblas_set_num_threads(int) __attribute__((weak));
    if (openblas_set_num_threads != nullptr)
    {
      int nt = (num_threads == 0) ? g_mil_state.cpu_info.logical_cores : static_cast<int>(num_threads);
      openblas_set_num_threads(nt);
    }
    break;
  }
  case MIL_BACKEND_BLIS:
  {
#ifdef _OPENMP
    int nt = (num_threads == 0) ? g_mil_state.cpu_info.logical_cores : static_cast<int>(num_threads);
    omp_set_num_threads(nt);
#endif
    break;
  }
  default:
    break;
  }

  return MIL_OK;
}

size_t mil_get_num_threads()
{
  if (!g_mil_state.initialized)
  {
    return 0;
  }

  if (g_mil_state.num_threads == 0)
  {
    return static_cast<size_t>(g_mil_state.cpu_info.logical_cores);
  }

  return g_mil_state.num_threads;
}

const char *mil_strerror(int error)
{
  switch (error)
  {
  case MIL_OK:
    return "Success";
  case MIL_ERR_NOT_INITIALIZED:
    return "MIL not initialized";
  case MIL_ERR_INVALID_ARG:
    return "Invalid argument";
  case MIL_ERR_NO_BACKEND:
    return "No BLAS backend available";
  case MIL_ERR_INTERNAL:
    return "Internal error";
  case MIL_ERR_ALLOCATION:
    return "Memory allocation failed";
  case MIL_ERR_UNSUPPORTED:
    return "Operation not supported";
  default:
    return "Unknown error";
  }
}

const char *mil_get_system_info()
{
  static char info_buffer[1024];

  if (!g_mil_state.initialized)
  {
    return "MIL not initialized";
  }

  std::snprintf(info_buffer, sizeof(info_buffer),
                "Microkernel Interface Layer (MIL) System Info:\n"
                "  Backend: %s\n"
                "  Threads: %zu\n"
                "  CPU: %d cores (%d logical)\n"
                "  Features: AVX=%d AVX2=%d AVX512=%d AMX=%d\n"
                "  Cache: L1D=%dKB L2=%dKB L3=%dKB\n"
                "  Prefetch: %s\n"
                "  Auto-tuning: %s\n",
                mil_backend_name(g_mil_state.active_backend),
                mil_get_num_threads(),
                g_mil_state.cpu_info.cores,
                g_mil_state.cpu_info.logical_cores,
                g_mil_state.cpu_info.avx,
                g_mil_state.cpu_info.avx2,
                g_mil_state.cpu_info.avx512,
                g_mil_state.cpu_info.amx,
                g_mil_state.cpu_info.l1d_kb,
                g_mil_state.cpu_info.l2_kb,
                g_mil_state.cpu_info.l3_kb,
                g_mil_state.enable_prefetch ? "Enabled" : "Disabled",
                g_mil_state.enable_auto_tuning ? "Enabled" : "Disabled");

  return info_buffer;
}