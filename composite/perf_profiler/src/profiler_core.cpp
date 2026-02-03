#include "profiler_api.h"
#include "profiler_papi.h"
#include "profiler_region.h"
#include <atomic>
#include <benchmark.h>
#include <config.h>
#include <iostream>
#include <jcore_hw_introspect.h>

using jcore::microbench::Sample;
using jcore::microbench::ScopedTimer;

namespace jcore::profiler {

static bool g_initialized = false;
static thread_local PAPIProfiler *tls_papi_profiler = nullptr;
static std::atomic<bool> g_papi_available{false};

bool Init() noexcept {
  if (g_initialized)
    return true;

  if (jcore_init() != JCORE_HW_OK)
    return false;

  // Initialize PAPI once globally
  InitPAPI();

  g_initialized = true;
  return true;
}

void Shutdown() noexcept {
  if (!g_initialized)
    return;

  // Clean up thread-local PAPI profiler
  if (tls_papi_profiler != nullptr) {
    delete tls_papi_profiler;
    tls_papi_profiler = nullptr;
  }

  jcore_shutdown();
  g_initialized = false;
}

void ProfileRegion(const std::string &name,
                   const std::function<void()> &fn) noexcept {
  if (!g_initialized || !fn)
    return;

  // Initialize thread-local PAPI profiler if not done yet
  if (g_papi_available.load() && tls_papi_profiler == nullptr) {
    tls_papi_profiler = new PAPIProfiler();
    if (!tls_papi_profiler->Init()) {
      delete tls_papi_profiler;
      tls_papi_profiler = nullptr;
    }
  }

  TimingSample ts{};
  RegionCounters counters{};

  // Start PAPI counters BEFORE work begins
  bool papi_started = false;
  if (tls_papi_profiler != nullptr) {
    papi_started = tls_papi_profiler->StartRegion(name);
    if (!papi_started) {
      std::cerr << "[PAPI] Failed to start region: " << name << "\n";
    }
  }

  // Time the function execution with ScopedTimer
  {
    ScopedTimer timer([&](const Sample &s) {
      ts.usec = s.usec;
      ts.cycles = s.cycles;
    });

    // Execute the actual work
    fn();

  } // ScopedTimer stops here

  // Stop PAPI counters AFTER work completes
  if (papi_started && tls_papi_profiler != nullptr) {
    if (!tls_papi_profiler->StopRegion(name)) {
      std::cerr << "[PAPI] Failed to stop region: " << name << "\n";
    } else {
      // Retrieve hardware counters
      if (!tls_papi_profiler->GetRegionCounters(name, counters)) {
        std::cerr << "[PAPI] Failed to get counters for: " << name << "\n";
      }
    }
  }

  // Record both timing and hardware counters
  RegionRegistry::Instance().Record(name, ts, counters);
}

} // namespace jcore::profiler