#include "ffm_cache_block.h"

#ifndef FFM_PREFETCH_H
#define FFM_PREFETCH_H

/**
 * @file ffm_prefetch.h
 * @brief Memory Prefetch Interface - Lightweight macros/intrinsics for cache-aware loading.
 *
 * GCC/Clang safe version: uses compile-time constants for __builtin_prefetch().
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* -------------------------------------------------------------------------- */
/* Compile-time constants for Prefetch Parameters */
/* -------------------------------------------------------------------------- */
#define FFM_PREFETCH_READ 0
#define FFM_PREFETCH_WRITE 1

#define FFM_PREFETCH_T0 3
#define FFM_PREFETCH_T1 2
#define FFM_PREFETCH_T2 1
#define FFM_PREFETCH_NTA 0

/* -------------------------------------------------------------------------- */
/* Prefetch Macro - Must use compile-time constants */
/* -------------------------------------------------------------------------- */
#define FFM_PREFETCH(addr, rw, locality) \
  __builtin_prefetch((addr), (rw), (locality))

  /* -------------------------------------------------------------------------- */
  /* Function Variants (compile-time constant specializations) */
  /* -------------------------------------------------------------------------- */

  /**
   * @brief Prefetch a memory block for read access (T0 locality).
   */
  static inline void ffm_prefetch_block_read_T0(const void *base, size_t size)
  {
    if (!base || size == 0)
      return;
    const unsigned char *p = (const unsigned char *)base;
    const size_t CACHE_LINE = 64;

    for (size_t i = 0; i < size; i += CACHE_LINE)
      __builtin_prefetch(p + i, FFM_PREFETCH_READ, FFM_PREFETCH_T0);
  }

  /**
   * @brief Prefetch a memory block for write access (T0 locality).
   */
  static inline void ffm_prefetch_block_write_T0(void *base, size_t size)
  {
    if (!base || size == 0)
      return;
    unsigned char *p = (unsigned char *)base;
    const size_t CACHE_LINE = 64;

    for (size_t i = 0; i < size; i += CACHE_LINE)
      __builtin_prefetch(p + i, FFM_PREFETCH_WRITE, FFM_PREFETCH_T0);
  }

  /**
   * @brief Prefetch a single address (for read).
   */
  static inline void ffm_prefetch_addr_read(const void *addr)
  {
    if (addr)
      __builtin_prefetch(addr, FFM_PREFETCH_READ, FFM_PREFETCH_T0);
  }

  /**
   * @brief Prefetch a single address (for write).
   */
  static inline void ffm_prefetch_addr_write(void *addr)
  {
    if (addr)
      __builtin_prefetch(addr, FFM_PREFETCH_WRITE, FFM_PREFETCH_T0);
  }

  /* -------------------------------------------------------------------------- */
  /* External Cache Utility (Tile Computation)                                  */
  /* -------------------------------------------------------------------------- */

  /**
   * @struct ffm_cache_info_t
   * @brief Describes cache-level information such as size and line size.
   *
   * Define this struct in your cache_info header/module.
   */
  // Forward declaration only; don't typedef recursively
  struct ffm_cache_info;

  /**
   * @brief Computes a recommended tile size (in elements) for a given cache level.
   *
   * @param info Pointer to cache information structure.
   * @param level Cache hierarchy level (e.g., 1=L1, 2=L2, etc.)
   * @param elem_size Size of one matrix element in bytes.
   * @param occupancy_fraction Fraction of cache to be used (e.g., 0.75 for 75% usage).
   * @return Recommended tile size (in number of elements).
   */
  size_t ffm_cache_compute_tile(ffm_cache_info_t *info,
                                int level,
                                size_t elem_size,
                                double occupancy_fraction);

#ifdef __cplusplus
}
#endif

#endif /* FFM_PREFETCH_H */
