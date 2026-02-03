#include "profiler_api.h"
#include "profiler_region.h"
#include "pool_manager.h"

#include <sstream>
#include <cstring>
#include <iomanip>
#include <mutex>

extern "C" void *ffm_malloc(size_t);

static pm_t *g_profiler_pool = nullptr;
static std::once_flag g_pool_init_flag;

static void InitProfilerPool()
{
  size_t pool_size = 16 * 1024 * 1024; // 16 MB
  size_t chunk_size = 4 * 1024; // 4 KB chunks
  pm_status_t status = pm_init(&g_profiler_pool, pool_size, chunk_size, 0, -1);
  if (status != PM_OK)
  {
    g_profiler_pool = nullptr;
  }
}

namespace jcore::profiler
{

  char *ExportJSON() noexcept
  {
    auto snapshot = RegionRegistry::Instance().Snapshot();

    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"profiler_version\": \"1.0\",\n";
    oss << "  \"regions\": [";

    bool first = true;
    for (const auto &kv : snapshot)
    {
      if (!first)
        oss << ",";
      first = false;

      const auto &stats = kv.second;

      // Calculate derived metrics
      double avg_usec = stats.calls > 0 ? stats.total_usec / stats.calls : 0.0;
      double ipc = stats.counters.cycles > 0
                       ? static_cast<double>(stats.counters.instructions) / stats.counters.cycles
                       : 0.0;

      // Check if this is a kernel dispatch region (contains "matmul_")
      bool is_kernel_region = (kv.first.find("matmul_") == 0);

      oss << "\n    {\n";
      oss << "      \"name\": \"" << kv.first << "\",\n";
      oss << "      \"type\": \"" << (is_kernel_region ? "kernel_dispatch" : "general") << "\",\n";
      oss << "      \"calls\": " << stats.calls << ",\n";
      oss << "      \"total_usec\": " << std::fixed << std::setprecision(2) << stats.total_usec << ",\n";
      oss << "      \"avg_usec\": " << std::fixed << std::setprecision(2) << avg_usec << ",\n";

      // Hardware counters
      oss << "      \"hardware_counters\": {\n";
      oss << "        \"l1_dcm\": " << stats.counters.l1_misses << ",\n";
      oss << "        \"l2_dcm\": " << stats.counters.l2_misses << ",\n";
      oss << "        \"l3_tcm\": " << stats.counters.l3_misses << ",\n";
      oss << "        \"instructions\": " << stats.counters.instructions << ",\n";
      oss << "        \"cycles\": " << stats.counters.cycles << ",\n";
      oss << "        \"ipc\": " << std::fixed << std::setprecision(3) << ipc << "\n";
      oss << "      }";

      // For kernel dispatch regions, calculate performance metrics
      if (is_kernel_region && stats.calls > 0)
      {
        // Extract dimensions from region name (e.g., "matmul_1024x1024x1024")
        size_t M = 0, N = 0, K = 0;
        if (sscanf(kv.first.c_str(), "matmul_%zux%zux%zu", &M, &N, &K) == 3)
        {
          double flops = 2.0 * M * N * K * stats.calls;
          double time_sec = stats.total_usec / 1e6;
          double gflops = flops / time_sec / 1e9;

          oss << ",\n";
          oss << "      \"performance\": {\n";
          oss << "        \"dimensions\": [" << M << ", " << N << ", " << K << "],\n";
          oss << "        \"total_flops\": " << std::scientific << std::setprecision(2) << flops << ",\n";
          oss << "        \"gflops\": " << std::fixed << std::setprecision(2) << gflops << "\n";
          oss << "      }";
        }
      }

      oss << "\n    }";
    }

    oss << "\n  ]\n";
    oss << "}";

    const std::string out = oss.str();

    // Initialize pool once
    std::call_once(g_pool_init_flag, InitProfilerPool);

    char *buf = nullptr;
    if (g_profiler_pool)
    {
      buf = static_cast<char *>(pm_alloc(g_profiler_pool));
    }
    else
    {
      buf = static_cast<char *>(ffm_malloc(out.size() + 1));
    }

    if (!buf)
      return nullptr;

    std::memcpy(buf, out.c_str(), out.size() + 1);
    return buf;
  }

  void ShutdownProfilerPool() noexcept
  {
    if (g_profiler_pool)
    {
      pm_shutdown(g_profiler_pool);
      g_profiler_pool = nullptr;
    }
  }

} // namespace jcore::profiler