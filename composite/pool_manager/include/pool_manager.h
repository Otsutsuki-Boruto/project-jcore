// include/pool_manager.h
#ifndef POOL_MANAGER_H_
#define POOL_MANAGER_H_

/*
 * Memory Pool Manager - FFM-compatible C API
 *
 * Features:
 *  - Pre-allocates a contiguous region (optionally via hugepages).
 *  - Carves fixed-size chunks for fast allocation/free.
 *  - Optional NUMA placement via ffm_set_numa_node / ffm_huge APIs.
 *  - Thread-safe (pthread mutex).
 *
 * Usage:
 *   pm_status_t st = pm_init(&pm, pool_bytes, chunk_bytes, use_hugepages_flag, numa_node);
 *   void *p = pm_alloc(&pm);
 *   pm_free(&pm, p);
 *   pm_shutdown(&pm);
 *
 * See pm_get_stats() for runtime info.
 *
 * Dependencies (headers):
 *   mem_wrapper.h, ffm_hugepage.h, jcore_hw_introspect.h, config.h
 *
 * All functions are re-entrant for different pm_t instances.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum
  {
    PM_OK = 0,
    PM_ERR_NO_MEMORY = -1,
    PM_ERR_INVALID_ARG = -2,
    PM_ERR_NOT_INITIALIZED = -3,
    PM_ERR_INTERNAL = -4,
    PM_ERR_ALREADY_INIT = -5,
  } pm_status_t;

  /* Opaque pool handle */
  typedef struct pm_t pm_t;

  /* Stats returned by pm_get_stats */
  typedef struct
  {
    size_t pool_bytes;   /* total pool size in bytes */
    size_t chunk_bytes;  /* user chunk size */
    size_t total_chunks; /* pool_bytes / chunk_bytes */
    size_t free_chunks;  /* currently free chunks */
    int using_hugepages; /* 1 if hugepages allocated, 0 otherwise */
    int numa_node;       /* configured numa node (-1 if none) */
  } pm_stats_t;

  /**
   * Initialize pool manager object.
   * - out: pointer to pm_t* (will be allocated by function). Caller must call pm_shutdown().
   * - pool_bytes: total contiguous region to reserve (rounded up to chunk_bytes multiple).
   * - chunk_bytes: fixed size returned on pm_alloc (>= sizeof(void*) and power-of-two recommended).
   * - use_hugepages: non-zero to try hugepage allocation via ffm_huge_alloc; fallback to ffm_malloc.
   * - numa_node: >=0 to pin to that numa node (if supported), -1 for none.
   *
   * Returns PM_OK on success. On failure *out is set to NULL.
   */
  pm_status_t pm_init(pm_t **out, size_t pool_bytes, size_t chunk_bytes,
                      int use_hugepages, int numa_node);

  /**
   * Allocate one chunk from pool. Returns pointer or NULL on failure (no memory).
   * Thread-safe relative to pm_free/pm_shutdown.
   */
  void *pm_alloc(pm_t *pm);
  /**
   * Expand an existing memory pool by allocating additional backing memory.
   * Returns PM_OK on success or PM_ERR_* on failure.
   */

  pm_status_t pm_expand(pm_t *pm, size_t additional_bytes);

  /**
   * Free a chunk previously returned from pm_alloc.
   * If ptr is NULL, no-op.
   * Returns PM_OK on success, error code otherwise.
   */
  pm_status_t pm_free(pm_t *pm, void *ptr);

  /**
   * Shutdown pool manager and free all resources.
   * Safe to call multiple times. After this pm pointer is invalid and must not be used.
   */
  void pm_shutdown(pm_t *pm);

  /**
   * Retrieve runtime statistics. Returns PM_OK on success.
   * If pm is NULL or out_stats is NULL returns PM_ERR_INVALID_ARG.
   */
  pm_status_t pm_get_stats(pm_t *pm, pm_stats_t *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* POOL_MANAGER_H_ */
