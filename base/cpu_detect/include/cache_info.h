#pragma once

#include <string>
#include <vector>

struct CacheInfo
{
  std::string level;
  std::string type;
  int size_kb;
};

// Read cache info from sysfs
std::vector<CacheInfo> read_cache_sysfs(int cpu = 0);

// Fallback L2 cache detection via CPUID
int cpuid_l2_cache_kb();
