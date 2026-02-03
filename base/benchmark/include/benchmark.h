// include/benchmark.h
#ifndef JC_CORE_MICROBENCH_BENCHMARK_H_
#define JC_CORE_MICROBENCH_BENCHMARK_H_

// Microbenchmark & Timer Utilities - Project JCore
// - High-resolution timers (clock_gettime preferred).
// - Optional cycle counter (rdtsc on x86/x86_64).
// - MicroBenchmark runner: warmup, repeated measurements, summary statistics.
// - RAII ScopedTimer
// - C-compatible wrapper for minimal FFM API integration.
//
// Design/Style:
// - Google C++ style guidelines for naming & comments.
// - Functions return error codes or throw std::runtime_error for truly fatal errors.
// - No dynamic global state that leaks; all resources are RAII-managed.

#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>
#include <string>

namespace jcore
{
  namespace microbench
  {

    // Basic struct carrying timing results in microseconds and cycles (if available).
    struct Sample
    {
      double usec;     // elapsed microseconds
      uint64_t cycles; // elapsed CPU cycles (0 if unavailable)
    };

    // Summary statistics returned after running multiple samples.
    struct Summary
    {
      std::size_t runs; // number of measurements (samples)
      double mean_usec;
      double median_usec;
      double stddev_usec;
      double min_usec;
      double max_usec;

      // cycles statistics (0 if CPU cycle counting not available)
      double mean_cycles;
      double median_cycles;
      double stddev_cycles;
      uint64_t min_cycles;
      uint64_t max_cycles;
    };

    // Query whether rdtsc cycle counter is available on this platform.
    bool HasCycleCounter() noexcept;

    // Read a high-resolution clock in microseconds (monotonic).
    // Returns true on success and stores value in out_usec; false on failure.
    bool NowUsec(double &out_usec) noexcept;

    // Read CPU cycles using rdtsc on supported platforms. If not available returns 0.
    uint64_t ReadCycles() noexcept;

    // RAII scoped timer: records duration between construction and destruction and calls
    // a provided callback with Sample. Non-throwing destructor.
    class ScopedTimer
    {
    public:
      // callback: void(const Sample&)
      explicit ScopedTimer(const std::function<void(const Sample &)> &callback) noexcept;
      ~ScopedTimer() noexcept;

      // Cancel the timer so the callback won't be invoked (useful in early returns).
      void Cancel() noexcept;

      // Non-copyable
      ScopedTimer(const ScopedTimer &) = delete;
      ScopedTimer &operator=(const ScopedTimer &) = delete;

    private:
      struct Impl;
      Impl *impl_;
    };

    // Microbenchmark runner options
    struct RunOptions
    {
      std::size_t warmup_iterations = 1; // number of warmup invocations before measurement
      std::size_t iterations = 1000;     // number of iterations per sample (aggregated work)
      std::size_t samples = 10;          // number of samples to collect
      bool capture_cycles = true;        // attempt to capture cycles if available
    };

    // Run a microbenchmark:
    // - func: void(std::size_t iter_index) invoked 'iterations' times per sample.
    // - options: run options
    // - out_samples: filled with per-sample timing information
    // Returns true on success, false on failure (e.g., func is empty).
    bool RunMicrobenchmark(const std::function<void(std::size_t)> &func,
                           const RunOptions &options,
                           std::vector<Sample> &out_samples) noexcept;

    // Compute summary statistics from samples. Returns true on success.
    bool Summarize(const std::vector<Sample> &samples, Summary &out_summary) noexcept;

    // Human-readable summary string (for logging).
    std::string SummaryToString(const Summary &s) noexcept;

    // Minimal C-compatible wrapper for FFM-like integration.
    //
    // Run a benchmark of a callback function pointer (void (*cb)(size_t)).
    // Returns 0 on success, non-zero on fatal error.
    // The results are written to 'out_summary' pointer if non-null.
    extern "C"
    {
      struct CRunOptions
      {
        size_t warmup_iterations;
        size_t iterations;
        size_t samples;
        int capture_cycles;
      };
      // Callback type: user-provided function performing one iteration. Must be noexcept (C ABI).
      typedef void (*bench_cb_t)(size_t idx);
      int ffm_run_benchmark(bench_cb_t cb, const CRunOptions *opts, Summary *out_summary);
    }

  } // namespace microbench
} // namespace jcore

#endif // JC_CORE_MICROBENCH_BENCHMARK_H_
