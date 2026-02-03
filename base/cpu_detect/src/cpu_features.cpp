#include "cpu_features.h"
#include <cpuid.h>
#include <immintrin.h>
#include <cstdint>

using u32 = uint32_t;
using u64 = uint64_t;

// -- CPUID helpers -----------------------------------------------------------
static inline void cpuid_count(u32 leaf, u32 subleaf, u32 &a, u32 &b, u32 &c, u32 &d)
{
  __cpuid_count(leaf, subleaf, a, b, c, d);
}
static inline void cpuid(u32 leaf, u32 &a, u32 &b, u32 &c, u32 &d)
{
  __cpuid(leaf, a, b, c, d);
}

// XGETBV: read extended control register
static inline u64 xgetbv(u32 index)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  return _xgetbv(index);
#else
  return 0ull;
#endif
}

// Detect CPU features
CPUFeatures detect_cpu_features()
{
  CPUFeatures f;
  u32 eax, ebx, ecx, edx;

  cpuid(1, eax, ebx, ecx, edx);
  bool osxsave = (ecx >> 27) & 1;
  bool avx_hw = (ecx >> 28) & 1;

  if (osxsave && avx_hw)
  {
    u64 xcr0 = xgetbv(0);
    if ((xcr0 & 0x6ULL) == 0x6ULL)
      f.avx = true;
  }

  cpuid_count(7, 0, eax, ebx, ecx, edx);
  bool avx2_hw = (ebx >> 5) & 1;
  bool avx512f_hw = (ebx >> 16) & 1;
  if (f.avx && avx2_hw)
    f.avx2 = true;

  if (avx512f_hw)
  {
    u64 xcr0 = xgetbv(0);
    if ((xcr0 & 0x1E6ULL) == 0x1E6ULL)
      f.avx512 = true;
  }

  // Intel AMX detection
  cpuid(0, eax, ebx, ecx, edx);
  if (eax >= 0x1D)
  {
    u32 a2, b2, c2, d2;
    cpuid_count(0x1D, 1, a2, b2, c2, d2);
    if ((a2 | b2 | c2 | d2) != 0)
      f.amx = true;
  }

  return f;
}
