#include "numa_info.h"
#include <numa.h>
#include <iostream>

void query_numa()
{
  if (numa_available() < 0)
  {
    std::cerr << "NUMA not available on this system.\n";
    return;
  }
  int nodes = numa_num_configured_nodes();
  int maxnode = numa_max_node();
  std::cout << "NUMA nodes (configured): " << nodes << ", max node: " << maxnode << "\n";

  for (int n = 0; n <= maxnode; ++n)
  {
    struct bitmask *bm = numa_allocate_cpumask();
    if (numa_node_to_cpus(n, bm) == 0)
    {
      std::cout << "Node " << n << " CPUs: ";
      bool any = false;
      for (unsigned i = 0; i < bm->size; ++i)
      {
        if (numa_bitmask_isbitset(bm, i))
        {
          if (any)
            std::cout << ",";
          std::cout << i;
          any = true;
        }
      }
      std::cout << "\n";
    }
    numa_free_cpumask(bm);
  }
}
