#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct cpu_info_t
  {
    int cores;         // Configured cores
    int logical_cores; // Online cores
    bool avx;
    bool avx2;
    bool avx512;
    bool amx;
    int l1d_kb;
    int l1i_kb;
    int l2_kb;
    int l3_kb;
    int numa_nodes; // Number of NUMA nodes
  } cpu_info_t;

  /**
   * Detect CPU features, cache info, cores, and NUMA nodes.
   * Returns a filled cpu_info_t struct.
   */
  cpu_info_t detect_cpu_info();

#ifdef __cplusplus
}
#endif
