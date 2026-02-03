/*
 * src/topology.c
 *
 * hwloc-based topology introspection and JSON builder.
 *
 * Fixes:
 *  - jcore_shutdown() has correct signature (int) to match header.
 *  - Avoids signed/unsigned mixing by using signed int indices and safe casts.
 *  - append_json() grows buffer dynamically (no fixed 16KB limit).
 *  - Defensive NULL checks before dereferencing hwloc objects.
 */

#define _GNU_SOURCE
#include "jcore_hw_introspect.h"
#include <hwloc.h>
#include <numa.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h> /* INT_MAX */

/* Single global topology instance (one definition). */
hwloc_topology_t g_topology = NULL;

/* Dynamic append helper: grows *buf as needed. Returns 0 on success, -1 on error. */
static int append_json(char **buf, size_t *cap, size_t *len, const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  int needed = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (needed < 0)
    return -1;

  size_t need_total = *len + (size_t)needed + 1;
  if (need_total > *cap)
  {
    size_t newcap = (*cap == 0) ? 1024 : *cap * 2;
    while (newcap < need_total)
      newcap *= 2;
    char *nb = realloc(*buf, newcap);
    if (!nb)
      return -1;
    *buf = nb;
    *cap = newcap;
  }

  va_start(ap, fmt);
  int written = vsnprintf(*buf + *len, *cap - *len, fmt, ap);
  va_end(ap);
  if (written < 0)
    return -1;
  *len += (size_t)written;
  return 0;
}

/* --------------------- Lifecycle (single definition) --------------------- */

int jcore_init(void)
{
  if (g_topology)
    return JCORE_HW_OK; /* already initialized */

  if (hwloc_topology_init(&g_topology) < 0)
  {
    fprintf(stderr, "[JCore] hwloc_topology_init failed\n");
    g_topology = NULL;
    return JCORE_HW_ERR_HWLOC_FAIL;
  }
  if (hwloc_topology_load(g_topology) < 0)
  {
    fprintf(stderr, "[JCore] hwloc_topology_load failed\n");
    hwloc_topology_destroy(g_topology);
    g_topology = NULL;
    return JCORE_HW_ERR_HWLOC_FAIL;
  }
  if (numa_available() < 0)
  {
    fprintf(stderr, "[JCore] Warning: NUMA not available on this system\n");
  }
  return JCORE_HW_OK;
}

int jcore_shutdown(void)
{
  if (!g_topology)
    return JCORE_HW_OK;
  hwloc_topology_destroy(g_topology);
  g_topology = NULL;
  return JCORE_HW_OK;
}

/* --------------------- Topology JSON builder --------------------- */

char *jcore_get_topology_json(void)
{
  if (!g_topology)
    return NULL;

  char *buf = NULL;
  size_t cap = 0, len = 0;

  unsigned num_packages = hwloc_get_nbobjs_by_type(g_topology, HWLOC_OBJ_PACKAGE);
  unsigned num_cores = hwloc_get_nbobjs_by_type(g_topology, HWLOC_OBJ_CORE);
  unsigned num_pus = hwloc_get_nbobjs_by_type(g_topology, HWLOC_OBJ_PU);
  unsigned num_numanodes = hwloc_get_nbobjs_by_type(g_topology, HWLOC_OBJ_NUMANODE);

  if (append_json(&buf, &cap, &len, "{\n") < 0)
    goto fail;
  if (append_json(&buf, &cap, &len,
                  "  \"packages\": %u,\n"
                  "  \"cores\": %u,\n"
                  "  \"logical_processors\": %u,\n"
                  "  \"numa_nodes\": %u,\n",
                  num_packages, num_cores, num_pus, num_numanodes) < 0)
    goto fail;

  /* processors array */
  if (append_json(&buf, &cap, &len, "  \"processors\": [\n") < 0)
    goto fail;

  for (unsigned pu = 0; pu < num_pus; ++pu)
  {
    hwloc_obj_t objpu = hwloc_get_obj_by_type(g_topology, HWLOC_OBJ_PU, pu);
    if (!objpu)
      continue; /* defensive */

    /* Safe ancestor lookups with NULL checks */
    hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(g_topology, HWLOC_OBJ_CORE, objpu);
    hwloc_obj_t pkg = hwloc_get_ancestor_obj_by_type(g_topology, HWLOC_OBJ_PACKAGE, objpu);
    hwloc_obj_t numa = hwloc_get_ancestor_obj_by_type(g_topology, HWLOC_OBJ_NUMANODE, objpu);

    /* Use signed int indices to avoid mixing signed/unsigned in ternary operator */
    int core_idx = -1;
    int pkg_idx = -1;
    int numa_idx = 0; /* default 0 for single-node clarity */

    if (core)
    {
      if (core->logical_index <= (unsigned)INT_MAX)
        core_idx = (int)core->logical_index;
      else
        core_idx = INT_MAX;
    }

    if (pkg)
    {
      if (pkg->logical_index <= (unsigned)INT_MAX)
        pkg_idx = (int)pkg->logical_index;
      else
        pkg_idx = INT_MAX;
    }

    if (numa)
    {
      if (numa->logical_index <= (unsigned)INT_MAX)
        numa_idx = (int)numa->logical_index;
      else
        numa_idx = INT_MAX;
    }

    /* Safe formatting: package/core may be -1 if unknown; numa_node defaults to 0 when not present */
    if (append_json(&buf, &cap, &len,
                    "    { \"logical_id\": %u, \"os_index\": %u, \"package\": %d, \"core\": %d, \"numa_node\": %d }%s\n",
                    pu,
                    (unsigned)objpu->os_index,
                    pkg_idx,
                    core_idx,
                    numa_idx,
                    (pu + 1 == num_pus) ? "" : ",") < 0)
      goto fail;
  }

  if (append_json(&buf, &cap, &len, "  ],\n") < 0)
    goto fail;

  /* NUMA nodes memory info (if any) */
  if (append_json(&buf, &cap, &len, "  \"numa_info\": [\n") < 0)
    goto fail;
  for (unsigned i = 0; i < num_numanodes; ++i)
  {
    hwloc_obj_t nn = hwloc_get_obj_by_type(g_topology, HWLOC_OBJ_NUMANODE, i);
    if (!nn)
      continue;
    unsigned long long mem = nn->total_memory ? nn->total_memory : 0ULL;
    if (append_json(&buf, &cap, &len,
                    "    { \"node\": %u, \"local_memory_bytes\": %llu }%s\n",
                    nn->logical_index, mem, (i + 1 == num_numanodes) ? "" : ",") < 0)
      goto fail;
  }
  if (append_json(&buf, &cap, &len, "  ],\n") < 0)
    goto fail;

  /* Embed CPU features JSON (from cpuid.c) */
  extern char *jcore_get_cpu_features_json(void);
  char *feat = jcore_get_cpu_features_json();
  if (feat)
  {
    if (append_json(&buf, &cap, &len, "  \"cpu_features\": %s\n", feat) < 0)
    {
      free(feat);
      goto fail;
    }
    free(feat);
  }
  else
  {
    if (append_json(&buf, &cap, &len, "  \"cpu_features\": null\n") < 0)
      goto fail;
  }

  if (append_json(&buf, &cap, &len, "}\n") < 0)
    goto fail;

  return buf;

fail:
  free(buf);
  return NULL;
}

/* FFM-safe free helper (matches header declaration) */
void jcore_free(void *ptr)
{
  free(ptr);
}
