#include "profiler_papi.h"
#include <iostream>

namespace jcore
{
  namespace profiler
  {

    PAPIProfiler::PAPIProfiler() noexcept {}

    PAPIProfiler::~PAPIProfiler() noexcept
    {
      Shutdown();
    }

    bool PAPIProfiler::Init()
    {
      // PAPI_library_init should be called once globally
      // This is just thread-local initialization (no-op for now)
      return true;
    }

    void PAPIProfiler::Shutdown() noexcept
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto &kv : regions_)
      {
        if (kv.second.event_set != PAPI_NULL)
        {
          // Stop if still running
          long long dummy[5] = {0};
          PAPI_stop(kv.second.event_set, dummy);
          PAPI_cleanup_eventset(kv.second.event_set);
          PAPI_destroy_eventset(&kv.second.event_set);
        }
      }
      regions_.clear();
    }

    bool PAPIProfiler::StartRegion(const std::string &region_name)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      RegionData &data = regions_[region_name];

      // Check if region is already active (prevent resource leak)
      if (data.event_set != PAPI_NULL)
      {
        std::cerr << "[PAPI] Warning: Region '" << region_name
                  << "' already active. Cleaning up previous instance.\n";
        long long dummy[5] = {0};
        PAPI_stop(data.event_set, dummy);
        PAPI_cleanup_eventset(data.event_set);
        PAPI_destroy_eventset(&data.event_set);
        data.event_set = PAPI_NULL;
      }

      // Create new event set
      if (PAPI_create_eventset(&data.event_set) != PAPI_OK)
      {
        std::cerr << "[PAPI] Failed to create event set for: " << region_name << "\n";
        return false;
      }

      // Add events: L1, L2, L3 cache misses, instructions, cycles
      int events[] = {PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L3_TCM, PAPI_TOT_INS, PAPI_TOT_CYC};
      if (PAPI_add_events(data.event_set, events, 5) != PAPI_OK)
      {
        std::cerr << "[PAPI] Failed to add events for: " << region_name << "\n";
        PAPI_destroy_eventset(&data.event_set);
        data.event_set = PAPI_NULL;
        return false;
      }

      // Start counting
      if (PAPI_start(data.event_set) != PAPI_OK)
      {
        std::cerr << "[PAPI] Failed to start counters for: " << region_name << "\n";
        PAPI_cleanup_eventset(data.event_set);
        PAPI_destroy_eventset(&data.event_set);
        data.event_set = PAPI_NULL;
        return false;
      }

      return true;
    }

    bool PAPIProfiler::StopRegion(const std::string &region_name)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = regions_.find(region_name);
      if (it == regions_.end())
      {
        std::cerr << "[PAPI] Region '" << region_name << "' not found\n";
        return false;
      }

      if (it->second.event_set == PAPI_NULL)
      {
        std::cerr << "[PAPI] Region '" << region_name << "' not started\n";
        return false;
      }

      long long values[5] = {0};
      if (PAPI_stop(it->second.event_set, values) != PAPI_OK)
      {
        std::cerr << "[PAPI] Failed to stop counters for: " << region_name << "\n";
        return false;
      }

      // Store the counter values
      it->second.l1_misses = values[0];
      it->second.l2_misses = values[1];
      it->second.l3_misses = values[2];
      it->second.instructions = values[3];
      it->second.cycles = values[4];

      // Clean up the event set
      PAPI_cleanup_eventset(it->second.event_set);
      PAPI_destroy_eventset(&it->second.event_set);
      it->second.event_set = PAPI_NULL;

      return true;
    }

    bool PAPIProfiler::GetRegionCounters(const std::string &region_name, RegionCounters &out_counters) const
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = regions_.find(region_name);
      if (it == regions_.end())
      {
        return false;
      }

      out_counters.l1_misses = it->second.l1_misses;
      out_counters.l2_misses = it->second.l2_misses;
      out_counters.l3_misses = it->second.l3_misses;
      out_counters.cycles = it->second.cycles;
      out_counters.instructions = it->second.instructions;

      return true;
    }

  } // namespace profiler
} // namespace jcore