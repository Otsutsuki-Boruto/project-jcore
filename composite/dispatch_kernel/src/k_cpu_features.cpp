// k_cpu_features.cpp

#include "cpu_features.h"
#include <cpuid.h> // GCC builtin for CPUID
#include <vector>

CPUFeatures detect_cpu_features() {
  CPUFeatures f{};
  unsigned int eax, ebx, ecx, edx;

  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    f.avx = (ecx & bit_AVX) != 0;
  }
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    f.avx2 = (ebx & bit_AVX2) != 0;
    f.avx512 = (ebx & bit_AVX512F) != 0;
    f.amx = (ebx & bit_AMX_TILE) != 0; // adjust if compiler defines differently
  }
  return f;
}