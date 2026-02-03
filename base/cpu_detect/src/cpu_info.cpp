#include "cpu_info.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef __linux__
#include <cpuid.h>
#include <immintrin.h>
#include <unistd.h>
#include <dirent.h>
#endif

typedef unsigned int u32;
typedef unsigned long long u64;

#ifdef __linux__
// ------------------------ CPUID / XGETBV helpers ------------------------
static inline void cpuid_count(u32 leaf, u32 subleaf, u32 *a, u32 *b, u32 *c, u32 *d)
{
  __cpuid_count(leaf, subleaf, *a, *b, *c, *d);
}
static inline void cpuid(u32 leaf, u32 *a, u32 *b, u32 *c, u32 *d)
{
  __cpuid(leaf, *a, *b, *c, *d);
}
static inline u64 xgetbv(u32 index)
{
#if defined(__GNUC__)
#if defined(__AVX__)
  return _xgetbv(index);
#else
  return 0ULL; // fallback if AVX not enabled
#endif
#else
  return 0ULL;
#endif
}

// ------------------------ Cache / NUMA helpers --------------------------
static int cpuid_l2_cache_kb()
{
  u32 eax, ebx, ecx, edx;
  cpuid(0x80000006, &eax, &ebx, &ecx, &edx);
  return (ecx >> 16) & 0xFFFF;
}

static void read_cache_sysfs(cpu_info_t *info)
{
  // Default to 0
  info->l1d_kb = info->l1i_kb = info->l2_kb = info->l3_kb = 0;

  const char *cpu_path = "/sys/devices/system/cpu/cpu0/cache";
  DIR *d = opendir(cpu_path);
  if (!d)
    return;

  struct dirent *ent;
  while ((ent = readdir(d)) != NULL)
  {
    if (strncmp(ent->d_name, "index", 5) != 0)
      continue;

    char path[512], type[16], level[8], size_str[16];
    snprintf(path, sizeof(path), "%s/%s/size", cpu_path, ent->d_name);
    FILE *f = fopen(path, "r");
    if (!f)
      continue;
    if (!fgets(size_str, sizeof(size_str), f))
    {
      fclose(f);
      continue;
    }
    fclose(f);

    int val_kb = 0;
    char last = size_str[strlen(size_str) - 2];
    if (last == 'K' || last == 'k')
      val_kb = atoi(size_str);
    else
      val_kb = atoi(size_str) / 1024;

    snprintf(path, sizeof(path), "%s/%s/type", cpu_path, ent->d_name);
    f = fopen(path, "r");
    if (!f)
      continue;
    if (!fgets(type, sizeof(type), f))
    {
      fclose(f);
      continue;
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s/%s/level", cpu_path, ent->d_name);
    f = fopen(path, "r");
    if (!f)
      continue;
    if (!fgets(level, sizeof(level), f))
    {
      fclose(f);
      continue;
    }
    fclose(f);

    if (strcmp(type, "Data") == 0 && atoi(level) == 1)
      info->l1d_kb = val_kb;
    if (strcmp(type, "Instruction") == 0 && atoi(level) == 1)
      info->l1i_kb = val_kb;
    if (strcmp(type, "Unified") == 0)
    {
      if (atoi(level) == 2)
        info->l2_kb = val_kb;
      if (atoi(level) == 3)
        info->l3_kb = val_kb;
    }
  }
  closedir(d);

  // Fallback for L2
  if (info->l2_kb == 0)
    info->l2_kb = cpuid_l2_cache_kb();
}

// ------------------------ NUMA helpers ----------------------------------
static int count_numa_nodes()
{
  int max_node = 0;
  DIR *d = opendir("/sys/devices/system/node/");
  if (!d)
    return 1;
  struct dirent *ent;
  while ((ent = readdir(d)) != nullptr)
  {
    if (strncmp(ent->d_name, "node", 4) == 0)
    {
      int n = atoi(ent->d_name + 4);
      if (n > max_node)
        max_node = n;
    }
  }
  closedir(d);
  return max_node + 1;
}

// ------------------------ CPU Feature detection -------------------------
static void detect_cpu_features(cpu_info_t *info)
{
  info->avx = info->avx2 = info->avx512 = info->amx = false;

  u32 eax, ebx, ecx, edx;

  // Basic features
  cpuid(1, &eax, &ebx, &ecx, &edx);
  bool osxsave = (ecx >> 27) & 1;
  bool avx_hw = (ecx >> 28) & 1;

  if (osxsave && avx_hw)
  {
    u64 xcr0 = xgetbv(0);
    if ((xcr0 & 0x6ULL) == 0x6ULL)
      info->avx = true;
  }

  // Extended features
  cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
  if (info->avx && ((ebx >> 5) & 1))
    info->avx2 = true;
  if ((ebx >> 16) & 1)
  {
    u64 xcr0 = xgetbv(0);
    if ((xcr0 & 0x1E6ULL) == 0x1E6ULL)
      info->avx512 = true;
  }

  // AMX detection (leaf 0x1D)
  cpuid(0, &eax, &ebx, &ecx, &edx);
  if (eax >= 0x1D)
  {
    cpuid_count(0x1D, 1, &eax, &ebx, &ecx, &edx);
    if (eax || ebx || ecx || edx)
      info->amx = true;
  }
}

#endif // __linux__

// ------------------------ Main API ---------------------------------------
cpu_info_t detect_cpu_info()
{
  cpu_info_t info = {};
#ifdef __linux__
  info.cores = sysconf(_SC_NPROCESSORS_CONF);
  info.logical_cores = sysconf(_SC_NPROCESSORS_ONLN);

  detect_cpu_features(&info);
  read_cache_sysfs(&info);
  info.numa_nodes = count_numa_nodes();
#else
  // Fallback defaults for non-Linux
  info.cores = 1;
  info.logical_cores = 1;
  info.avx = info.avx2 = info.avx512 = info.amx = false;
  info.l1d_kb = info.l1i_kb = info.l2_kb = info.l3_kb = 0;
  info.numa_nodes = 1;
#endif
  return info;
}
