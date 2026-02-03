#include "profiler_region.h"
#include "profiler_papi.h"
#include <iostream>
#include <atomic>

namespace jcore::profiler
{

  // Thread-local PAPI profiler for thread safety
  static thread_local PAPIProfiler *tls_papi_profiler = nullptr;
  static std::atomic<bool> g_papi_available{false};

  RegionRegistry &RegionRegistry::Instance()
  {
    static RegionRegistry inst;
    return inst;
  }

  bool InitPAPI() noexcept
  {
    static std::once_flag init_flag;
    static bool init_success = false;

    std::call_once(init_flag, []()
                   {
      // Try to initialize PAPI library once globally
      if (PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT) {
        init_success = true;
        g_papi_available.store(true);
      } else {
        std::cerr << "[PAPI] Library initialization failed - hardware counters disabled\n";
        init_success = false;
      } });

    // Initialize thread-local profiler instance if PAPI is available
    if (init_success && tls_papi_profiler == nullptr)
    {
      tls_papi_profiler = new PAPIProfiler();
      if (!tls_papi_profiler->Init())
      {
        delete tls_papi_profiler;
        tls_papi_profiler = nullptr;
        return false;
      }
    }

    return init_success;
  }

  // Original signature for backward compatibility
  void RegionRegistry::Record(const std::string &name,
                              const TimingSample &sample) noexcept
  {
    // Call the new overload with empty counters
    RegionCounters empty_counters{};
    Record(name, sample, empty_counters);
  }

  // New implementation that accepts hardware counters
  void RegionRegistry::Record(const std::string &name,
                              const TimingSample &sample,
                              const RegionCounters &counters) noexcept
  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto &entry = stats_[name];
    entry.calls++;
    entry.total_usec += sample.usec;

    // Accumulate hardware counters
    entry.counters.l1_misses += counters.l1_misses;
    entry.counters.l2_misses += counters.l2_misses;
    entry.counters.l3_misses += counters.l3_misses;
    entry.counters.cycles += counters.cycles;
    entry.counters.instructions += counters.instructions;
  }

  std::unordered_map<std::string, RegionStats>
  RegionRegistry::Snapshot() const noexcept
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

} // namespace jcore::profiler