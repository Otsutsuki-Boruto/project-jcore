#ifndef MEM_WRAPPER_H
#define MEM_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * Backend selection.
   * You can compile with -DUSE_JEMALLOC or -DUSE_MEMKIND to enable
   * the corresponding code paths. If neither is present,
   * the wrapper falls back to libc malloc.
   */
  typedef enum ffm_backend_t
  {
    FFM_BACKEND_AUTO = 0,   // pick best available at init time
    FFM_BACKEND_MALLOC = 1, // standard malloc/free
    FFM_BACKEND_JEMALLOC = 2,
    FFM_BACKEND_MEMKIND = 3
  } ffm_backend_t;

  /**
   * Status codes returned by ffm API.
   */
  typedef enum ffm_status_t
  {
    FFM_OK = 0,
    FFM_ERR_NO_MEMORY = 1,
    FFM_ERR_INVALID_ARG = 2,
    FFM_ERR_NOT_INITIALIZED = 3,
    FFM_ERR_UNSUPPORTED = 4,
    FFM_ERR_INTERNAL = 5,
    FFM_ERR_INIT = 6
  } ffm_status_t;

  /**
   * Initialize the allocator wrapper. Must be called before other APIs.
   * Call ffm_shutdown() before program exit to release any backend resources.
   * Pass FFM_BACKEND_AUTO to auto-detect supported backend.
   */
  ffm_status_t ffm_init(ffm_backend_t backend);

  /**
   * Shutdown and cleanup.
   * After this, other ffm_* calls are invalid until ffm_init is called again.
   */
  void ffm_shutdown(void);

  /**
   * Allocate memory (like malloc). Returns pointer or NULL on failure.
   */
  void *ffm_malloc(size_t size);

  /**
   * Allocate zero-initialized memory (like calloc).
   */
  void *ffm_calloc(size_t nmemb, size_t size);

  /**
   * Allocate aligned memory.
   * Alignment must be a power of two and multiple of sizeof(void*).
   */
  void *ffm_aligned_alloc(size_t alignment, size_t size);

  /**
   * Free memory returned by ffm_malloc/ffm_aligned_alloc/ffm_calloc.
   */
  void ffm_free(void *ptr);

  /**
   * Reallocate memory. Behavior matches realloc.
   */
  void *ffm_realloc(void *ptr, size_t new_size);

  /**
   * Set preferred NUMA node for subsequent allocations.
   * If libnuma not available, returns FFM_ERR_UNSUPPORTED.
   */
  ffm_status_t ffm_set_numa_node(int node);

  /**
   * Get current backend in use.
   */
  ffm_backend_t ffm_get_backend(void);

  /**
   * Get human-readable string for status code.
   */
  const char *ffm_status_str(ffm_status_t s);

#ifdef __cplusplus
}
#endif

#endif // MEM_WRAPPER_H
