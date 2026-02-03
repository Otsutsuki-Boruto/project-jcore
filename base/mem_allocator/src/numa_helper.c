/*
 * NUMA helper implementation.
 * - if libnuma present (detected via compile-time symbol USE_NUMA), uses numa_alloc_onnode and numa_free
 * - otherwise functions return -1 / NULL to indicate unsupported.
 */

#include "numa_helper.h"
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

static int g_numa_available = 0;
static int g_numa_node = -1; // -1 = no affinity

int numa_helper_init(void)
{
#ifdef USE_NUMA
  if (numa_available() < 0)
  {
    g_numa_available = 0;
    return -1;
  }
  g_numa_available = 1;
  g_numa_node = -1;
  return 0;
#else
  (void)printf("[numa_helper] libnuma not compiled in; NUMA functions unavailable.\n");
  return -1;
#endif
}

int numa_helper_set_node(int node)
{
#ifdef USE_NUMA
  if (!g_numa_available)
    return -1;
  if (node < 0)
  {
    g_numa_node = -1;
    return 0;
  }
  if (node >= numa_num_configured_nodes())
    return -1;
  g_numa_node = node;
  return 0;
#else
  (void)node;
  return -1;
#endif
}

void *numa_helper_alloc_on_node(size_t size, int node)
{
#ifdef USE_NUMA
  if (!g_numa_available)
    return NULL;
  if (node < 0)
    node = g_numa_node; // use global if unspecified
  if (node >= 0)
  {
    void *p = numa_alloc_onnode(size, node);
    if (!p)
    {
      errno = ENOMEM;
      return NULL;
    }
    return p;
  }
  return malloc(size);
#else
  (void)size;
  (void)node;
  return NULL;
#endif
}

void numa_helper_free(void *ptr, size_t size)
{
#ifdef USE_NUMA
  if (!g_numa_available)
    return;
  if (ptr == NULL)
    return;
  numa_free(ptr, size);
#else
  (void)ptr;
  (void)size;
#endif
}