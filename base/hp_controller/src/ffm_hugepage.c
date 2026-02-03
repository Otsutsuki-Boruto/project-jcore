// Implementation of the FFM Huge Page Controller
#define _GNU_SOURCE // Needed to expose madvise and MADV_HUGEPAGE
#include "ffm_hugepage.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif

// Internal structure definition
struct ffm_huge_region
{
  void *addr;           // mapped address
  size_t size;          // requested size
  int used_map_hugetlb; // boolean: was MAP_HUGETLB used
};

static int ffm_inited = 0;
static size_t ffm_kernel_page = 0;   // regular page size
static size_t ffm_hugepage_size = 0; // huge page size (2MB expected)

// Forward declarations
static int detect_hugepage_size(void);
static int try_map_hugetlb(size_t size, void **out);
static int try_map_anonymous(size_t size, void **out);

/* helper: best-effort detection of transparent hugepages.
 * kept for diagnostics; may not be used on all builds.
 * Marked unused to silence -Wunused-function under -Wall -Wextra.
 */
static __attribute__((unused)) int is_transparent_hugepage_enabled(void)
{
  FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
  if (!f)
    return 0;
  char buf[256];
  if (!fgets(buf, sizeof(buf), f))
  {
    fclose(f);
    return 0;
  }
  fclose(f);
  return (strstr(buf, "madvise") != NULL) || (strstr(buf, "always") != NULL);
}

int ffm_huge_init(void)
{
  if (ffm_inited)
    return 0;
  ffm_kernel_page = (size_t)sysconf(_SC_PAGESIZE);
  if (ffm_kernel_page == 0)
    return -ENOTSUP;
  if (detect_hugepage_size() != 0)
  {
    // proceed but ffm_hugepage_size may be 0
  }
  ffm_inited = 1;
  return 0;
}

void ffm_huge_shutdown(void)
{
  ffm_inited = 0;
  ffm_kernel_page = 0;
  ffm_hugepage_size = 0;
}

// returns 0 on success (out set), negative errno on failure
static int try_map_anonymous(size_t size, void **out)
{
  // Round size up to page size
  size_t pagesz = ffm_kernel_page ? ffm_kernel_page : 4096;
  size_t alloc = (size + pagesz - 1) & ~(pagesz - 1);

  void *addr = mmap(NULL, alloc, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED)
    return -errno;
  *out = addr;
  return 0;
}

static int try_map_hugetlb(size_t size, void **out)
{
#ifdef MAP_HUGETLB
  size_t hsz = ffm_hugepage_size ? ffm_hugepage_size : (2ULL * 1024 * 1024);
  size_t alloc = (size + hsz - 1) & ~(hsz - 1);
  void *addr = mmap(NULL, alloc, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  if (addr == MAP_FAILED)
    return -errno;
  *out = addr;
  return 0;
#else
  (void)size;
  (void)out;
  return -ENOSYS;
#endif
}

// Read /proc/meminfo to detect Hugepagesize (best-effort)
static int detect_hugepage_size(void)
{
  FILE *f = fopen("/proc/meminfo", "r");
  if (!f)
    return -errno;
  char line[256];
  size_t hsize = 0;
  while (fgets(line, sizeof(line), f))
  {
    unsigned long val;
    if (sscanf(line, "Hugepagesize: %lu kB", &val) == 1)
    {
      hsize = val * 1024UL;
      break;
    }
  }
  fclose(f);
  if (hsize)
    ffm_hugepage_size = hsize;
  return hsize ? 0 : -ENOENT;
}

ffm_huge_region_t *ffm_huge_alloc(size_t size, int prefer_transparent)
{
  if (!ffm_inited)
  {
    int r = ffm_huge_init();
    if (r)
    {
      errno = -r;
      return NULL;
    }
  }
  if (size == 0)
  {
    errno = EINVAL;
    return NULL;
  }

  // Allocate container
  ffm_huge_region_t *r = calloc(1, sizeof(*r));
  if (!r)
  {
    errno = ENOMEM;
    return NULL;
  }

  void *addr = NULL;
  int rc = 0;

  // 1) Prefer transparent hugepages via madvise on anonymous mapping
  if (prefer_transparent)
  {
    rc = try_map_anonymous(size, &addr);
    if (rc == 0)
    {
      // try to madvise
      if (madvise(addr, (size + ffm_kernel_page - 1) & ~(ffm_kernel_page - 1), MADV_HUGEPAGE) != 0)
      {
        // madvise failed; still usable but note: fall back attempt below
      }
      r->addr = addr;
      r->size = (size + ffm_kernel_page - 1) & ~(ffm_kernel_page - 1);
      r->used_map_hugetlb = 0;
      return r;
    }
    // if mapping failed, fall through to MAP_HUGETLB attempt
  }

  // 2) Try MAP_HUGETLB (requires kernel reserved huge pages or CAP_SYS_ADMIN)
  rc = try_map_hugetlb(size, &addr);
  if (rc == 0)
  {
    r->addr = addr;
    // round to hugepage size
    size_t hsz = ffm_hugepage_size ? ffm_hugepage_size : (2ULL * 1024 * 1024);
    r->size = (size + hsz - 1) & ~(hsz - 1);
    r->used_map_hugetlb = 1;
    return r;
  }

  // 3) Last resort: anonymous mapping without madvise
  rc = try_map_anonymous(size, &addr);
  if (rc == 0)
  {
    r->addr = addr;
    r->size = (size + ffm_kernel_page - 1) & ~(ffm_kernel_page - 1);
    r->used_map_hugetlb = 0;
    return r;
  }

  // All attempts failed
  free(r);
  errno = -rc;
  return NULL;
}

int ffm_huge_touch(ffm_huge_region_t *r, unsigned long pattern)
{
  if (!r || !r->addr || r->size == 0)
    return -EINVAL;
  // Write 8-byte words across the memory to fault pages in
  uint64_t *p = (uint64_t *)r->addr;
  size_t words = r->size / sizeof(uint64_t);
  for (size_t i = 0; i < words; ++i)
  {
    p[i] = (uint64_t)pattern ^ (uint64_t)i;
  }
  // Optionally, use msync to synchronize to RAM? Not necessary.
  return 0;
}

int ffm_huge_free(ffm_huge_region_t *rgn)
{
  if (!rgn)
    return -EINVAL;
  int rc = 0;
  if (rgn->addr && rgn->size)
  {
    if (munmap(rgn->addr, rgn->size) != 0)
      rc = -errno;
  }
  free(rgn);
  return rc;
}

int ffm_huge_is_hugetlb(ffm_huge_region_t *r)
{
  if (!r)
    return 0;
  return r->used_map_hugetlb ? 1 : 0;
}

void *ffm_huge_ptr(ffm_huge_region_t *r)
{
  if (!r)
    return NULL;
  return r->addr;
}
size_t ffm_huge_size(ffm_huge_region_t *r)
{
  if (!r)
    return 0;
  return r->size;
}