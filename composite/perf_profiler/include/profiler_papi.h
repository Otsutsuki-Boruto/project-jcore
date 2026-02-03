#pragma once
#include "profiler_types.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <papi.h>

namespace jcore
{
  namespace profiler
  {

    /**
     * HPC-grade telemetry using PAPI hardware counters.
     *
     * Provides:
     *  - Cache misses (L1, L2, L3)
     *  - Memory accesses / bandwidth
     *  - CPU cycles
     */
    class PAPIProfiler
    {
    public:
      PAPIProfiler() noexcept;
      ~PAPIProfiler() noexcept;

      // Initialize PAPI library. Returns true if successful.
      bool Init();

      // Shutdown PAPI library.
      void Shutdown() noexcept;

      // Start counters for a region
      bool StartRegion(const std::string &region_name);

      // Stop counters for a region and store measurements
      bool StopRegion(const std::string &region_name);

      // Retrieve counters for a region
      bool GetRegionCounters(const std::string &region_name, RegionCounters &out_counters) const;

    private:
      struct RegionData
      {
        long long l1_misses = 0;
        long long l2_misses = 0;
        long long l3_misses = 0;
        long long cycles = 0;
        long long instructions = 0;
        int event_set = PAPI_NULL;
      };

      std::unordered_map<std::string, RegionData> regions_;
      mutable std::mutex mutex_;
    };

  } // namespace profiler
} // namespace jcore
