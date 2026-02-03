#pragma once
#include <cstddef>
#include <cstdint>

namespace jcore::profiler
{

  struct TimingSample
  {
    double usec = 0.0;
    uint64_t cycles = 0;
  };

  struct RegionCounters
  {
    long long l1_misses = 0;
    long long l2_misses = 0;
    long long l3_misses = 0;
    long long cycles = 0;
    long long instructions = 0;
  };

  struct RegionStats
  {
    std::size_t calls = 0;
    double total_usec = 0.0;
    RegionCounters counters;
  };
} // namespace jcore::profiler
