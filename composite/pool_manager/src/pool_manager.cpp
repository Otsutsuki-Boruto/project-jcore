// src/pool_manager.cpp
/*
 * pool_manager.cpp
 *
 *  - Per-thread caches for O(1) fast-path alloc/free
 *  - NUMA-aware placement (best-effort via jcore_hw_introspect)
 *  - Pool expansion (allocate additional regions when depleted)
 *  - Uses FFM allocator wrapper and FFM hugepage APIs
 *
 * Build: g++ -std=c++17 -O3 -mavx -fopenmp ... (see project compile command)
 *
 * Notes:
 *  -
 *  - The pool hands out fixed-size chunks. All chunks are carve-outs of contiguous
 *    regions allocated via ffm_malloc() or ffm_huge_alloc().
 *  - Per-thread caches are thread-local singly-linked stacks; the global
 *    free-list is protected by a mutex and used as fallback.
 *
 * Safety:
 *  - pm_free() validates pointers belong to a known region and are aligned.
 *  - pm_shutdown() waits for no concurrent access by acquiring central lock,
 *    but the caller must ensure no outstanding references remain or use
 *    higher-level synchronization (recommended).
 */

#define _POSIX_C_SOURCE 200809L
#include "pool_manager.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <numa.h>

/* Base component headers (assumed present in include path) */
#include "mem_wrapper.h"                          /* ffm_* APIs */
#include "ffm_hugepage.h"                         /* ffm_huge_* APIs */
#include "jcore_hw_introspect.h"                  /* jcore_hw_introspect_* APIs (best-effort) */
#include "../../base/config_env/include/config.h" /* project config - maybe included later. Uncommenting causes include errors*/
#include "jcore_hw_introspect.h"

/* ----------------------------- design constants --------------------------- */
/* Default per-thread cache size (number of chunks to cache per thread) */
#ifndef PM_DEFAULT_THREAD_CACHE
#define PM_DEFAULT_THREAD_CACHE 64
#endif

/* When expanding, allocate this many bytes for each new region by default:
   use original region_size (the initial pool size) to keep regions uniform. */
#define PM_EXPANSION_REGIONS 1

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define THREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL __thread
#endif

/* ----------------------------- internal types ----------------------------- */

/* free node stored in each chunk */
typedef struct free_node
{
  struct free_node *next;
} free_node_t;

/* track each allocated region so we can free it on shutdown */
typedef struct region_node
{
  void *ptr;           /* base pointer */
  size_t size;         /* size in bytes */
  int using_hugepages; /* 1 if allocated via hugepages */
  void *huge_handle;   /* store ffm_huge_region_t* if hugepages */
  struct region_node *next;
} region_node_t;

struct pm_t
{
  int scaling_mode; /* 0 = normal fast-path, 1 = NUMA-aware / fine-grained */

#if defined(HAS_SPINLOCKS)
  pthread_spinlock_t regional_locks[MAX_NUMA_NODES];
#endif

  /* configuration */
  size_t chunk_bytes;
  size_t region_size; /* size for each region (pool_bytes rounded up) */

  /* runtime */
  pthread_mutex_t central_lock;   /* protects central free list and regions */
  free_node_t *central_free_list; /* central freelist head */
  size_t total_chunks;            /* across all regions */
  size_t free_chunks_count;       /* central + per-thread caches aggregated (approx) */

  /* region tracking */
  region_node_t *regions; /* linked list of all allocated regions */

  /* flags */
  int using_hugepages_initial; /* initial request */
  int initialized;
  int numa_node; /* configured numa node (-1 none) */
};

/* ---------- per-thread cache (thread-local) ----------
 * Simple design: each thread keeps a small singly-linked list of chunks.
 * This list is not shared and does not need atomics; operations are O(1).
 *
 * We use C11's _Thread_local storage for portability in C.
 */
typedef struct tcache
{
  free_node_t *head;
  size_t count;
  size_t max_count;
} tcache_t;

static THREAD_LOCAL tcache_t pm_tcache = {NULL, 0, PM_DEFAULT_THREAD_CACHE};
/* ------------------------- helper utilities ------------------------------- */

/* Round up x to multiple of a */
static inline size_t round_up(size_t x, size_t a)
{
  return ((x + a - 1) / a) * a;
}

/* Push a node to a (non-atomic) singly-linked list head */
static inline void push_node(free_node_t **head, free_node_t *node)
{
  node->next = *head;
  *head = node;
}

/* Pop a node from a singly-linked list head; returns NULL if empty */
static inline free_node_t *pop_node(free_node_t **head)
{
  free_node_t *n = *head;
  if (n)
  {
    *head = n->next;
    n->next = NULL;
  }
  return n;
}

/* Push N nodes from a contiguous region into central freelist under lock.
   Caller must hold central_lock. */
static void push_region_chunks_to_central(pm_t *pm, void *base, size_t bytes)
{
  uint8_t *cur = (uint8_t *)base;
  size_t n = bytes / pm->chunk_bytes;
  for (size_t i = 0; i < n; ++i)
  {
    free_node_t *node = (free_node_t *)cur;
    /* push onto central freelist */
    node->next = pm->central_free_list;
    pm->central_free_list = node;
    cur += pm->chunk_bytes;
    __atomic_fetch_add(&pm->free_chunks_count, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&pm->total_chunks, 1, __ATOMIC_RELAXED);
  }
}

/* Validate a pointer belongs to tracked regions and is aligned to chunk boundary.
   Returns 1 if valid, 0 otherwise. */
static int pointer_belongs_to_regions(pm_t *pm, void *ptr)
{
  if (!pm || !ptr)
    return 0;
  uintptr_t p = (uintptr_t)ptr;

  for (region_node_t *r = pm->regions; r != NULL; r = r->next)
  {
    uintptr_t base = (uintptr_t)r->ptr;
    if (p >= base && p < base + r->size)
    {
      size_t offset = (size_t)(p - base);
      if (offset % pm->chunk_bytes == 0)
        return 1;
      else
        return 0;
    }
  }
  return 0;
}

/* Attempt to get system NUMA info and pick a node when numa_node == -1.
   This is best-effort: if jcore_hw_introspect does not provide nodes, leave -1.
   We don't fail init if introspection fails. */
/* Attempt to get system NUMA info and pick a node when numa_node == -1.
   Uses libnuma directly instead of jcore_hw_introspect. */
static int pm_detect_default_numa_node(void)
{
#ifdef USE_NUMA
  if (numa_available() < 0)
  {
    fprintf(stderr, "[INFO] NUMA not available on this system.\n");
    return -1;
  }

  int nodes = numa_num_configured_nodes();
  int maxnode = numa_max_node();

  if (nodes > 0)
  {
    /* Simple policy: choose the first node (0).
       You can later improve this by CPU affinity awareness. */
    fprintf(stderr, "[INFO] NUMA detected: %d nodes (max node %d). Using node 0.\n",
            nodes, maxnode);
    return 0;
  }
  else
  {
    fprintf(stderr, "[INFO] NUMA reported zero nodes, defaulting to -1.\n");
    return -1;
  }
#else
  return -1;
#endif
}

/* Allocate a new region of size = pm->region_size. Pin to pm->numa_node if set
   and attempt hugepages if using_hugepages_initial is true. Returns 0 on success. */
static pm_status_t pm_allocate_region(pm_t *pm, int try_huge)
{
  if (!pm)
    return PM_ERR_INVALID_ARG;

  void *region_ptr = NULL;
  ffm_huge_region_t *hreg = NULL;
  int using_huge = 0;

  /* ---------- Try hugepage allocation (if enabled) ---------- */
  if (try_huge && pm->using_hugepages_initial)
  {
    hreg = ffm_huge_alloc(pm->region_size, 0 /* prefer transparent */);
    if (hreg)
    {
      region_ptr = ffm_huge_ptr(hreg);
      if (region_ptr)
      {
        using_huge = 1;
      }
      else
      {
        /* failed to map region */
        ffm_huge_free(hreg);
        hreg = NULL;
      }
    }
  }

  /* ---------- Fallback: normal NUMA-aware allocation ---------- */
  if (!region_ptr)
  {
    if (pm->numa_node >= 0)
    {
      if (ffm_set_numa_node(pm->numa_node) != FFM_OK)
      {
        /* NUMA pinning failed – proceed without binding */
        pm->numa_node = -1;
      }
    }

    region_ptr = ffm_malloc(pm->region_size);
    if (!region_ptr)
      return PM_ERR_NO_MEMORY;

    using_huge = 0;
    hreg = NULL;
  }

  /* ---------- Register region metadata ---------- */
  region_node_t *rn = (region_node_t *)ffm_malloc(sizeof(region_node_t));
  if (!rn)
  {
    /* clean up allocation */
    if (using_huge)
    {
      ffm_huge_free(hreg);
    }
    else
    {
      ffm_free(region_ptr);
    }
    return PM_ERR_NO_MEMORY;
  }

  rn->ptr = region_ptr;
  rn->size = pm->region_size;
  rn->using_hugepages = using_huge;
  rn->huge_handle = using_huge ? (void *)hreg : NULL;
  rn->next = pm->regions;
  pm->regions = rn;

  /* ---------- Carve region into chunks ---------- */
  push_region_chunks_to_central(pm, region_ptr, pm->region_size);

  return PM_OK;
}

/* -------------------------- public API implementations --------------------- */

pm_status_t pm_init(pm_t **out, size_t pool_bytes, size_t chunk_bytes,
                    int use_hugepages, int numa_node)
{
  if (out == NULL || pool_bytes == 0 || chunk_bytes < sizeof(void *))
  {
    return PM_ERR_INVALID_ARG;
  }

  *out = NULL;

  /* initialize ffm wrapper (idempotent) */
  if (ffm_init(FFM_BACKEND_AUTO) != FFM_OK)
  {
    /* if ffm_init is not available or fails, we still attempt to proceed but return error */
    return PM_ERR_INTERNAL;
  }

  pm_t *pm = (pm_t *)ffm_malloc(sizeof(pm_t));
  if (!pm)
    return PM_ERR_NO_MEMORY;
  memset(pm, 0, sizeof(*pm));

  pm->chunk_bytes = chunk_bytes;
  pm->region_size = round_up(pool_bytes, chunk_bytes);
  pm->central_free_list = NULL;
  pm->regions = NULL;
  pm->total_chunks = 0;
  pm->free_chunks_count = 0;
  pm->using_hugepages_initial = use_hugepages ? 1 : 0;
  pm->initialized = 0;
  pm->numa_node = numa_node;

  if (pthread_mutex_init(&pm->central_lock, NULL) != 0)
  {
    ffm_free(pm);
    return PM_ERR_INTERNAL;
  }

  /* If user asked for automatic NUMA selection (numa_node == -1), try to detect */
  if (pm->numa_node < 0)
  {
    int detected = pm_detect_default_numa_node();
    if (detected >= 0)
      pm->numa_node = detected;
  }

  /* --- Scalability mode detection --- */
  long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (num_cores > 32 || (pm->numa_node >= 0))
  {
    pm->scaling_mode = 1; /* enable advanced scalability optimizations */
#if defined(HAS_SPINLOCKS)
    for (int i = 0; i < MAX_NUMA_NODES; ++i)
      pthread_spin_init(&pm->regional_locks[i], PTHREAD_PROCESS_PRIVATE);
#endif
  }
  else
  {
    pm->scaling_mode = 0; /* keep lightweight fast-path */
  }

  /* Allocate initial region(s) */
  if (pthread_mutex_lock(&pm->central_lock) != 0)
  {
    pthread_mutex_destroy(&pm->central_lock);
    ffm_free(pm);
    return PM_ERR_INTERNAL;
  }

  pm_status_t st = pm_allocate_region(pm, 1 /*try huge*/);
  pthread_mutex_unlock(&pm->central_lock);

  if (st != PM_OK)
  {
    pthread_mutex_destroy(&pm->central_lock);
    ffm_free(pm);
    return st;
  }

  pm->initialized = 1;
  *out = pm;
  return PM_OK;
}

void *pm_alloc(pm_t *pm)
{
  if (pm == NULL)
    return NULL;
  if (!pm->initialized)
    return NULL;

  /* Fast path: try thread-local cache */
  if (pm_tcache.head != NULL)
  {
    free_node_t *n = pm_tcache.head;
    pm_tcache.head = n->next;
    pm_tcache.count--;
    pm->free_chunks_count--;
    return (void *)n;
  }

  /* Slow path: take from central freelist under lock */
  if (pthread_mutex_lock(&pm->central_lock) != 0)
  {
    return NULL;
  }

  free_node_t *node = pm->central_free_list;
  if (node)
  {
    pm->central_free_list = node->next;
    pm->free_chunks_count--;

    pthread_mutex_unlock(&pm->central_lock);

    /* Give to thread-local cache for faster subsequent ops: keep one for return */
    return (void *)node;
  }

  /* No chunk available: attempt expansion */
  pm_status_t st = pm_allocate_region(pm, pm->using_hugepages_initial);
  if (st == PM_OK)
  {
    /* After expansion, central_free_list should have been populated.
       Pop one element to return. */
    node = pm->central_free_list;
    if (node)
    {
      pm->central_free_list = node->next;
      pm->free_chunks_count--;
      pthread_mutex_unlock(&pm->central_lock);
      return (void *)node;
    }
    else
    {
      /* strange: expansion succeeded but no nodes -> fail */
      pthread_mutex_unlock(&pm->central_lock);
      return NULL;
    }
  }
  else
  {
    /* expansion failed */
    pthread_mutex_unlock(&pm->central_lock);
    return NULL;
  }
}

pm_status_t pm_free(pm_t *pm, void *ptr)
{
  if (pm == NULL)
    return PM_ERR_INVALID_ARG;
  if (ptr == NULL)
    return PM_OK;
  if (!pm->initialized)
    return PM_ERR_NOT_INITIALIZED;

  /* Validate pointer belongs to known region and aligned */
  if (!pointer_belongs_to_regions(pm, ptr))
  {
    return PM_ERR_INVALID_ARG;
  }

  free_node_t *node = (free_node_t *)ptr;

  /* Fast path: return to thread-local cache if not full */
  if (pm_tcache.count < pm_tcache.max_count)
  {
    node->next = pm_tcache.head;
    pm_tcache.head = node;
    pm_tcache.count++;
    pm->free_chunks_count++;

    return PM_OK;
  }

  /* Slow path: push to central freelist under lock */
  if (pthread_mutex_lock(&pm->central_lock) != 0)
  {
    return PM_ERR_INTERNAL;
  }
  node->next = pm->central_free_list;
  pm->central_free_list = node;
  pm->free_chunks_count++;

  pthread_mutex_unlock(&pm->central_lock);
  return PM_OK;
}

/**
 * Expand an existing pool by at least `additional_bytes`.
 *
 * - Allocates one or more new regions (each of size pm->region_size)
 * - Pushes their chunks onto the central free list.
 * - Safe under concurrency (protected by pm->central_lock).
 * - Partial success is allowed (some regions allocated, some failed).
 */
pm_status_t pm_expand(pm_t *pm, size_t additional_bytes)
{
  if (pm == NULL)
    return PM_ERR_INVALID_ARG;
  if (!pm->initialized)
    return PM_ERR_NOT_INITIALIZED;
  if (additional_bytes == 0)
    return PM_ERR_INVALID_ARG;

  size_t region_sz = pm->region_size;
  if (region_sz == 0)
    return PM_ERR_INTERNAL;

  /* Determine number of regions to allocate (rounded up) */
  size_t regions_needed = (additional_bytes + region_sz - 1) / region_sz;
  if (regions_needed == 0)
    regions_needed = 1;

  if (pthread_mutex_lock(&pm->central_lock) != 0)
    return PM_ERR_INTERNAL;

  if (pm->scaling_mode && pm->numa_node >= 0)
  {
    /* Allocate regions interleaved across NUMA nodes */
    int node_count = numa_num_configured_nodes();
    if (node_count <= 0)
      node_count = 1; /* fallback for single-node systems */

    for (size_t i = 0; i < regions_needed; ++i)
    {
      int target_node = (pm->numa_node + (int)i) % node_count;
      ffm_set_numa_node(target_node);
      pm_allocate_region(pm, pm->using_hugepages_initial);
    }

    pthread_mutex_unlock(&pm->central_lock);
    return PM_OK;
  }

  int added = 0;
  for (size_t i = 0; i < regions_needed; ++i)
  {
    pm_status_t st = pm_allocate_region(pm, pm->using_hugepages_initial);
    if (st == PM_OK)
    {
      added++;
    }
    else
    {
      /* stop at first allocation failure (partial success allowed) */
      break;
    }
  }

  pthread_mutex_unlock(&pm->central_lock);

  if (added == 0)
    return PM_ERR_NO_MEMORY;
  return PM_OK;
}

void pm_shutdown(pm_t *pm)
{
  if (pm == NULL)
    return;

  if (!pm->initialized)
  {
    ffm_free(pm);
    return;
  }

  pthread_mutex_lock(&pm->central_lock);

  region_node_t *r = pm->regions;
  while (r)
  {
    region_node_t *next = r->next;

    if (r->using_hugepages)
    {
#if defined(FFM_HAVE_HUGE_FREE_BY_PTR)
      ffm_huge_free_by_ptr(r->ptr);
#else
      /* fallback: free by handle or pointer */
      if (r->huge_handle)
        ffm_huge_free((ffm_huge_region_t *)r->huge_handle);
      else if (r->ptr)
        ffm_huge_free((ffm_huge_region_t *)r->ptr);
#endif
    }
    else
    {
      if (r->ptr)
        ffm_free(r->ptr);
    }

    ffm_free(r);
    r = next;
  }

  pm->regions = NULL;
  pm->central_free_list = NULL;
  pm->free_chunks_count = 0;
  pm->total_chunks = 0;
  pm->initialized = 0;

  pthread_mutex_unlock(&pm->central_lock);
  pthread_mutex_destroy(&pm->central_lock);
  ffm_free(pm);
}

pm_status_t pm_get_stats(pm_t *pm, pm_stats_t *out_stats)
{
  if (pm == NULL || out_stats == NULL)
    return PM_ERR_INVALID_ARG;
  if (!pm->initialized)
    return PM_ERR_NOT_INITIALIZED;

  if (pthread_mutex_lock(&pm->central_lock) != 0)
    return PM_ERR_INTERNAL;
  out_stats->pool_bytes = pm->region_size;
  out_stats->chunk_bytes = pm->chunk_bytes;
  out_stats->total_chunks = pm->total_chunks;
  out_stats->free_chunks = pm->free_chunks_count;
  out_stats->using_hugepages = pm->using_hugepages_initial ? 1 : 0;
  out_stats->numa_node = pm->numa_node;
  pthread_mutex_unlock(&pm->central_lock);

  return PM_OK;
}
