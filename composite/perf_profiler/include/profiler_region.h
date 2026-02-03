#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include "profiler_types.h"

namespace jcore::profiler
{
  class RegionRegistry
  {
  public:
    static RegionRegistry &Instance();

    // Keep original signature for compatibility
    void Record(const std::string &name, const TimingSample &sample) noexcept;

    // New overload that accepts hardware counters
    void Record(const std::string &name,
                const TimingSample &sample,
                const RegionCounters &counters) noexcept;

    std::unordered_map<std::string, RegionStats> Snapshot() const noexcept;

  private:
    RegionRegistry() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, RegionStats> stats_;
  };

  // Helper to initialize PAPI (called internally)
  bool InitPAPI() noexcept;

} // namespace jcore::profiler