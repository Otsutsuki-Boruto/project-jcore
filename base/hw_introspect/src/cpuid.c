/*
 * src/cpuid.c
 *
 * CPU feature detection helpers. Returns JSON via jcore_get_cpu_features_json().
 *
 * Notes:
 *  - Uses GCC's <cpuid.h> helpers where available.
 *  - Variables that receive CPUID outputs are initialized to avoid
 *    -Wmaybe-uninitialized warnings during optimized inlining.
 */

#include "jcore_hw_introspect.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

/* Detect CPU features and produce a malloc()-ed JSON string.
 * Caller must free with jcore_free().
 */
char *jcore_get_cpu_features_json(void)
{
  /* Initialize registers to zero to avoid maybe-uninitialized warnings. */
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  char vendor[13] = {0};

  /* Attempt to read vendor string (CPUID leaf 0). If not supported,
   * fall back to "unknown".
   */
#if defined(__x86_64__) || defined(__i386__)
  if (__get_cpuid(0, &eax, &ebx, &ecx, &edx))
  {
    /* ebx, edx, ecx now contain the vendor ID string (4 bytes each). */
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
  }
  else
  {
    strcpy(vendor, "unknown");
  }
#else
  strcpy(vendor, "unknown");
#endif

  /* Feature flags (conservative checks). Defaults to 0 if not available. */
  int avx = 0, avx2 = 0, avx512f = 0, amx = 0;
  unsigned int logical = 0, cores_per_pkg = 0;

#if defined(__x86_64__) || defined(__i386__)
  unsigned int maxleaf = __get_cpuid_max(0, NULL);

  if (maxleaf >= 1)
  {
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    int osxsave = (ecx >> 27) & 1;
    int avx_bit = (ecx >> 28) & 1;
    avx = (osxsave && avx_bit) ? 1 : 0;
    logical = (ebx >> 16) & 0xff;
  }

  if (maxleaf >= 7)
  {
    /* extended features in leaf 7, subleaf 0 */
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    avx2 = ((ebx >> 5) & 1) ? 1 : 0;
    avx512f = ((ebx >> 16) & 1) ? 1 : 0;
  }

  /* AMX detection (Intel-specific leaf 0x1E subleaf 1 is one source) */
  /* Check only if leaf supported to avoid UB */
  if (maxleaf >= 0x1E)
  {
    __cpuid_count(0x1E, 1, eax, ebx, ecx, edx);
    /* conservative checks of EBX (implementation-defined encoding) */
    amx = ((ebx & 0x1) || (ebx & 0x2)) ? 1 : 0;
  }

  /* cores per package: attempt leaf 4 subleaf 0 */
  if (maxleaf >= 4)
  {
    __cpuid_count(4, 0, eax, ebx, ecx, edx);
    if (eax)
      cores_per_pkg = (((eax >> 26) & 0x3f) + 1);
  }
#endif

  /* Build JSON string (malloc'd). */
  size_t cap = 512;
  char *buf = (char *)malloc(cap);
  if (!buf)
    return NULL;

  int r = snprintf(buf, cap,
                   "{\n"
                   "  \"vendor\": \"%s\",\n"
                   "  \"logical_processors\": %u,\n"
                   "  \"cores_per_package\": %u,\n"
                   "  \"features\": {\n"
                   "    \"avx\": %d,\n"
                   "    \"avx2\": %d,\n"
                   "    \"avx512f\": %d,\n"
                   "    \"amx\": %d\n"
                   "  }\n"
                   "}\n",
                   vendor,
                   logical,
                   cores_per_pkg,
                   avx, avx2, avx512f, amx);

  if (r < 0)
  {
    free(buf);
    return NULL;
  }

  /* If truncated, allocate larger buffer and try again. */
  if ((size_t)r >= cap)
  {
    cap = (size_t)r + 1;
    char *nb = (char *)realloc(buf, cap);
    if (!nb)
    {
      free(buf);
      return NULL;
    }
    buf = nb;
    snprintf(buf, cap,
             "{\n"
             "  \"vendor\": \"%s\",\n"
             "  \"logical_processors\": %u,\n"
             "  \"cores_per_package\": %u,\n"
             "  \"features\": {\n"
             "    \"avx\": %d,\n"
             "    \"avx2\": %d,\n"
             "    \"avx512f\": %d,\n"
             "    \"amx\": %d\n"
             "  }\n"
             "}\n",
             vendor,
             logical,
             cores_per_pkg,
             avx, avx2, avx512f, amx);
  }

  return buf;
}
