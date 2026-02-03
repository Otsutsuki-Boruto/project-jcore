#pragma once
/*
 * jcore_hw_introspect.h
 *
 * Public API for the JCore Hardware Introspection Layer.
 *
 * Keep declarations here; implementations live in topology.c and other modules.
 */

#ifdef __cplusplus
extern "C"
{
#endif

  /* Return codes */

  static const int JCORE_HW_OK = 0;
  static const int JCORE_HW_ERR_INIT_FAILED = 1;
  static const int JCORE_HW_ERR_NOT_INITIALIZED = 2;
  static const int JCORE_HW_ERR_OUT_OF_MEMORY = 3;
  static const int JCORE_HW_ERR_HWLOC_FAIL = 4;

  /* Lifecycle - implemented in topology.c (single definition). */
  int jcore_init(void);
  int jcore_shutdown(void);

  /* Topology / features JSON producers (malloc'ed strings, caller frees with jcore_free). */
  char *jcore_get_topology_json(void);
  char *jcore_get_cpu_features_json(void);

  /* FFM-safe free for allocated buffers */
  void jcore_free(void *ptr);

#ifdef __cplusplus
}
#endif
