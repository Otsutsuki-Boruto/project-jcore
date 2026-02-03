// FFM-compatible Huge Page Controller API

#ifndef FFM_HUGEPAGE_H
#define FFM_HUGEPAGE_H

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Opaque handle for allocated huge page region
  typedef struct ffm_huge_region ffm_huge_region_t;

  // Initialize the hugepage subsystem. Call once at program startup.
  // Returns 0 on success, negative errno on failure.
  int ffm_huge_init(void);

  // Shutdown the subsystem and free internal resources. Safe to call multiple times.
  void ffm_huge_shutdown(void);

  // Allocate a region of `size` bytes suitable for 2MB huge pages.
  // If `prefer_transparent` != 0, the function will attempt madvise(MADV_HUGEPAGE) first
  // and fall back to MAP_HUGETLB (requires reserved hugepages and privileges).
  // Returns pointer to ffm_huge_region_t on success, NULL on failure (errno set).
  ffm_huge_region_t *ffm_huge_alloc(size_t size, int prefer_transparent);

  // Touch the allocated memory to fault pages in. If `pattern` != 0 each word will be
  // filled with the 64-bit pattern. Returns 0 on success, negative errno on failure.
  int ffm_huge_touch(ffm_huge_region_t *r, unsigned long pattern);

  // Free the region and release resources. After this call the pointer is invalid.
  // Returns 0 on success, negative errno on failure.
  int ffm_huge_free(ffm_huge_region_t *r);

  int ffm_huge_is_hugetlb(ffm_huge_region_t *r);

  // Retrieve pointer and size for direct access.
  void *ffm_huge_ptr(ffm_huge_region_t *r);
  size_t ffm_huge_size(ffm_huge_region_t *r);

#ifdef __cplusplus
}
#endif

#endif // FFM_HUGEPAGE_H