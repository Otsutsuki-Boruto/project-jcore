#pragma once
#include <cstdint>

struct CPUFeatures
{
  bool avx = false;
  bool avx2 = false;
  bool avx512 = false;
  bool amx = false;
};

// Detect CPU features: AVX/AVX2/AVX-512/AMX
CPUFeatures detect_cpu_features();
