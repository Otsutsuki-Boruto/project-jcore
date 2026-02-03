/*
 * ffm_api.c
 *
 * Public FFM-facing wrappers. This file MUST NOT implement jcore_init()
 * or jcore_shutdown() (those are implemented in topology.c only).
 *
 * ffm_api exposes a stable API for foreign callers (e.g. Java FFM).
 */

#include "jcore_hw_introspect.h"
#include <numa.h>
#include <stdio.h>

/* Forward declarations (implemented in topology.c) */
int jcore_init(void);
int jcore_shutdown(void);
char *jcore_get_topology_json(void);
char *jcore_get_cpu_features_json(void);
void jcore_free(void *ptr);

/* FFM-style thin wrappers (identical, kept for clarity and extension) */

int jcore_init_ffm(void)
{
  /* Detect NUMA presence (optional info) */
  if (numa_available() < 0)
  {
    fprintf(stderr, "[JCore] Warning: NUMA not available\n");
  }
  return jcore_init();
}

int jcore_shutdown_ffm(void)
{
  return jcore_shutdown();
}

/* The above wrappers are optional. FFM clients can call the canonical
 * jcore_* functions directly. We provide these wrappers to make it explicit
 * that this file is the FFM boundary (no duplicated implementation).
 */
