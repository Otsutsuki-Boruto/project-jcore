// composite/tuning_cache/include/cached_autotuner.h
#ifndef JCORE_CACHED_AUTOTUNER_H_
#define JCORE_CACHED_AUTOTUNER_H_

// Cached AutoTuner - Tuning Result Cache System Integration
//
// Purpose:
//   Wraps Adaptive Kernel AutoTuner with persistent caching.
//   Automatically caches tuning results and avoids re-benchmarking.
//
// Usage:
//   CachedAutoTuner tuner;
//   tuner.Init();
//   const char* best = tuner.SelectKernel(M, N, K, threads, tile_size);
//   tuner.Shutdown();
//
// This is the PRIMARY interface for the Tuning Result Cache System.
// It integrates adaptive_tuner.h with persistent caching automatically.

#include "adaptive_tuner.h"
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif

  // Status codes
  typedef enum
  {
    CAT_OK = 0,
    CAT_ERR_NOT_INITIALIZED,
    CAT_ERR_INVALID_ARG,
    CAT_ERR_NO_MEMORY,
    CAT_ERR_CACHE_FAILURE,
    CAT_ERR_TUNER_FAILURE,
    CAT_ERR_INTERNAL
  } cat_status_t;

  // Cache format
  typedef enum
  {
    CAT_FORMAT_BINARY = 0,
    CAT_FORMAT_JSON = 1
  } cat_format_t;

  // Configuration
  typedef struct
  {
    char cache_dir[512];   // Cache directory path
    cat_format_t format;   // Storage format
    size_t max_entries;    // Max cached entries (0=unlimited)
    int validate_hardware; // Validate HW signature (0/1)
    int auto_save;         // Auto-save on insert (0/1)
    int force_benchmark;   // Always benchmark, ignore cache (0/1)
  } cat_config_t;

  // Statistics
  typedef struct
  {
    size_t total_entries;
    size_t cache_hits;
    size_t cache_misses;
    size_t benchmarks_run;
    double hit_rate;
  } cat_stats_t;

  // Opaque handle
  typedef struct cat_handle_t cat_handle_t;

  // Initialize cached autotuner with default config
  cat_status_t cat_init(cat_handle_t **out_handle);

  // Initialize with custom config
  cat_status_t cat_init_with_config(cat_handle_t **out_handle, const cat_config_t *config);

  // Shutdown and cleanup
  void cat_shutdown(cat_handle_t *handle);

  // Select best kernel for given workload (main function)
  // Returns kernel name or NULL on error
  // Checks cache first, benchmarks if miss, stores result
  const char *cat_select_kernel(cat_handle_t *handle,
                                size_t M, size_t N, size_t K,
                                size_t threads, size_t tile_size);

  // Force re-benchmark and update cache
  cat_status_t cat_force_benchmark(cat_handle_t *handle,
                                   size_t M, size_t N, size_t K,
                                   size_t threads, size_t tile_size,
                                   char *out_kernel, size_t out_len);

  // Get statistics
  cat_status_t cat_get_stats(cat_handle_t *handle, cat_stats_t *out_stats);

  // Clear cache
  cat_status_t cat_clear_cache(cat_handle_t *handle);

  // Save cache to disk
  cat_status_t cat_save_cache(cat_handle_t *handle);

  // Load cache from disk
  cat_status_t cat_load_cache(cat_handle_t *handle);

  // Export cache to file
  cat_status_t cat_export_cache(cat_handle_t *handle, const char *filepath, cat_format_t format);

  // Import cache from file
  cat_status_t cat_import_cache(cat_handle_t *handle, const char *filepath);

  // Get status string
  const char *cat_status_str(cat_status_t s);

  // Initialize default config
  void cat_config_init_default(cat_config_t *config);

#ifdef __cplusplus
}
#endif

#endif // JCORE_CACHED_AUTOTUNER_H_