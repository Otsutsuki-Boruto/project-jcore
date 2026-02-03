#ifndef NUMA_HELPER_H
#define NUMA_HELPER_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * Initialize NUMA helper. Returns 0 on success, -1 on failure or if libnuma not present.
   */
  int numa_helper_init(void);

  /**
   * Set allocation node. If node < 0, disables node pinning.
   * Returns 0 on success, -1 on failure or unsupported.
   */
  int numa_helper_set_node(int node);

  /**
   * Allocate size bytes on specific node. Returns pointer or NULL.
   * If node < 0, behaves like regular malloc.
   */
  void *numa_helper_alloc_on_node(size_t size, int node);

  /**
   * Free memory allocated with numa_helper_alloc_on_node.
   */
  void numa_helper_free(void *ptr, size_t size);

#ifdef __cplusplus
}
#endif

#endif // NUMA_HELPER_H