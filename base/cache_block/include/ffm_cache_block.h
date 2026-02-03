// FFM-compatible Cache Blocking / Tiling Utility
// Computes L1/L2/L3-optimal block sizes for matrices (matrix multiply tiling).
// Portable best-effort implementation: reads cache sizes from sysfs
// (/sys/devices/system/cpu/cpu0/cache/index*) when available, otherwise
// falls back to sensible defaults. Follows C good-style and is compatible
// with the FFM API for inclusion in Project JCore.

#ifndef FFM_CACHE_BLOCK_H
#define FFM_CACHE_BLOCK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Opaque handle for cache info
  typedef struct ffm_cache_info ffm_cache_info_t;

  // Initialize cache info by probing sysfs. Returns pointer on success, NULL on failure.
  // The caller must free with ffm_cache_free().
  ffm_cache_info_t *ffm_cache_init(void);

  // Free cache info
  void ffm_cache_free(ffm_cache_info_t *info);

  // Print human-readable cache info to stdout
  void ffm_cache_print(ffm_cache_info_t *info);

  // Compute an optimal square block dimension (tile) for a given cache level.
  // level: 1 for L1, 2 for L2, 3 for L3. elem_size: bytes per matrix element (e.g., 8 for double).
  // occupancy_fraction: fraction of cache to use (0.0 < f <= 1.0), typical 0.6..0.9
  // This returns >=1 on success, 0 on error (invalid args or unknown level).
  size_t ffm_cache_compute_tile(ffm_cache_info_t *info, int level, size_t elem_size, double occupancy_fraction);

#ifdef __cplusplus
}
#endif

#endif // FFM_CACHE_BLOCK_H
