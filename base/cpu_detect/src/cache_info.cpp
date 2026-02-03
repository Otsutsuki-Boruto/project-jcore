#include "cache_info.h"
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <cpuid.h>
#include <iostream>
#include <stdint.h>

using u32 = uint32_t;

// Read sysfs cache info
std::vector<CacheInfo> read_cache_sysfs(int cpu)
{
  std::vector<CacheInfo> out;
  std::ostringstream base;
  base << "/sys/devices/system/cpu/cpu" << cpu << "/cache";
  DIR *d = opendir(base.str().c_str());
  if (!d)
    return out;

  struct dirent *ent;
  while ((ent = readdir(d)) != nullptr)
  {
    std::string name(ent->d_name);
    if (name.rfind("index", 0) != 0)
      continue;

    std::string path = base.str() + "/" + name + "/size";
    std::ifstream f(path);
    if (!f)
      continue;
    std::string size;
    std::getline(f, size);

    std::ifstream t(base.str() + "/" + name + "/type");
    std::string type;
    if (t)
      std::getline(t, type);

    std::ifstream l(base.str() + "/" + name + "/level");
    std::string lvl;
    if (l)
      std::getline(l, lvl);

    int valKB = 0;
    if (!size.empty())
    {
      if (size.back() == 'K' || size.back() == 'k')
        valKB = std::stoi(size.substr(0, size.size() - 1));
      else
        valKB = std::stoi(size) / 1024;
    }

    out.push_back({lvl, type, valKB});
  }
  closedir(d);
  return out;
}

// Fallback L2 cache via CPUID
int cpuid_l2_cache_kb()
{
  u32 eax, ebx, ecx, edx;
  __cpuid(0x80000006, eax, ebx, ecx, edx);
  return (ecx >> 16) & 0xffff;
}
