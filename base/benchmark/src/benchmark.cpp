// src/benchmark.cpp
#include "benchmark.h"

#include <chrono>
#include <ctime>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <atomic>
#include <thread>

#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include <time.h>
#endif

namespace jcore
{
  namespace microbench
  {

// ----------------------------- Cycle counter (rdtsc) -----------------------------
#if defined(__i386__) || defined(__x86_64__)
    static inline uint64_t rdtsc_raw() noexcept
    {
      unsigned int hi = 0u, lo = 0u;
#if defined(__x86_64__)
      __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
#else
      __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
#endif
      return (static_cast<uint64_t>(hi) << 32) | lo;
    }
    static const bool kHaveRdtsc = true;
#else
    static inline uint64_t rdtsc_raw() noexcept { return 0ull; }
    static const bool kHaveRdtsc = false;
#endif

    bool HasCycleCounter() noexcept
    {
      return kHaveRdtsc;
    }

    uint64_t ReadCycles() noexcept
    {
      if (!kHaveRdtsc)
        return 0ull;
      // Use serializing instruction to improve accuracy when possible.
#if defined(__x86_64__) || defined(__i386__)
      // rdtsc is not fully serializing; use cpuid before/after to order (costly but more correct).
      unsigned int eax, ebx, ecx, edx;
      eax = 0;
      __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));
      uint64_t t = rdtsc_raw();
      return t;
#else
      return rdtsc_raw();
#endif
    }

    // ----------------------------- High-resolution clock -----------------------------
    bool NowUsec(double &out_usec) noexcept
    {
#if defined(CLOCK_MONOTONIC_RAW)
      const clockid_t clk = CLOCK_MONOTONIC_RAW;
#elif defined(CLOCK_MONOTONIC)
      const clockid_t clk = CLOCK_MONOTONIC;
#else
      // fallback to chrono
      try
      {
        using namespace std::chrono;
        out_usec = static_cast<double>(duration_cast<duration<double, std::micro>>(steady_clock::now().time_since_epoch()).count());
        return true;
      }
      catch (...)
      {
        return false;
      }
#endif

      struct timespec ts;
      if (clock_gettime(clk, &ts) != 0)
      {
        // fallback to steady_clock
        try
        {
          using namespace std::chrono;
          out_usec = static_cast<double>(duration_cast<duration<double, std::micro>>(steady_clock::now().time_since_epoch()).count());
          return true;
        }
        catch (...)
        {
          return false;
        }
      }
      out_usec = static_cast<double>(static_cast<double>(ts.tv_sec) * 1e6 + static_cast<double>(ts.tv_nsec) / 1e3);
      return true;
    }

    // ----------------------------- ScopedTimer implementation -----------------------------
    struct ScopedTimer::Impl
    {
      std::function<void(const Sample &)> callback;
      double start_usec;
      uint64_t start_cycles;
      bool cancelled;

      Impl(const std::function<void(const Sample &)> &cb)
          : callback(cb), start_usec(0.0), start_cycles(0ull), cancelled(false)
      {
        NowUsec(start_usec);
        if (kHaveRdtsc)
          start_cycles = ReadCycles();
      }
    };

    ScopedTimer::ScopedTimer(const std::function<void(const Sample &)> &callback) noexcept
        : impl_(nullptr)
    {
      if (!callback)
      {
        // No-op timer (but keep object valid)
        impl_ = nullptr;
        return;
      }
      try
      {
        impl_ = new Impl(callback);
      }
      catch (...)
      {
        impl_ = nullptr;
      }
    }

    ScopedTimer::~ScopedTimer() noexcept
    {
      if (!impl_)
        return;
      if (impl_->cancelled)
      {
        delete impl_;
        impl_ = nullptr;
        return;
      }
      Sample s;
      double end_usec = 0.0;
      if (!NowUsec(end_usec))
      {
        // As a last resort set to zero elapsed
        s.usec = 0.0;
      }
      else
      {
        s.usec = end_usec - impl_->start_usec;
      }
      if (kHaveRdtsc)
      {
        uint64_t end_cycles = ReadCycles();
        s.cycles = (end_cycles >= impl_->start_cycles) ? (end_cycles - impl_->start_cycles) : 0ull;
      }
      else
      {
        s.cycles = 0ull;
      }
      // Ensure callback doesn't throw.
      try
      {
        impl_->callback(s);
      }
      catch (...)
      {
        // swallow exceptions - destructor must not throw
      }
      delete impl_;
      impl_ = nullptr;
    }

    void ScopedTimer::Cancel() noexcept
    {
      if (!impl_)
        return;
      impl_->cancelled = true;
    }

    // ----------------------------- Statistics & utilities -----------------------------
    static inline double mean(const std::vector<double> &v) noexcept
    {
      if (v.empty())
        return 0.0;
      double s = 0.0;
      for (double x : v)
        s += x;
      return s / static_cast<double>(v.size());
    }
    static inline double stddev(const std::vector<double> &v, double mu) noexcept
    {
      if (v.size() < 2)
        return 0.0;
      double s = 0.0;
      for (double x : v)
      {
        double d = x - mu;
        s += d * d;
      }
      return std::sqrt(s / static_cast<double>(v.size()));
    }
    static inline double median(std::vector<double> v) noexcept
    {
      if (v.empty())
        return 0.0;
      std::sort(v.begin(), v.end());
      std::size_t n = v.size();
      if (n % 2 == 1)
        return v[n / 2];
      return 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    // ----------------------------- Microbenchmark runner -----------------------------
    bool RunMicrobenchmark(const std::function<void(std::size_t)> &func,
                           const RunOptions &options,
                           std::vector<Sample> &out_samples) noexcept
    {
      out_samples.clear();
      if (!func)
      {
        std::cerr << "RunMicrobenchmark: func is empty" << std::endl;
        return false;
      }
      if (options.iterations == 0 || options.samples == 0)
      {
        std::cerr << "RunMicrobenchmark: invalid iterations/samples" << std::endl;
        return false;
      }

      // Warmup phase
      for (std::size_t w = 0; w < options.warmup_iterations; ++w)
      {
        try
        {
          for (std::size_t i = 0; i < options.iterations; ++i)
            func(i);
        }
        catch (...)
        {
          std::cerr << "RunMicrobenchmark: exception during warmup" << std::endl;
          return false;
        }
        // hint to CPU that we're warming up (small sleep to let caches settle)
        std::this_thread::yield();
      }

      // Measurement samples
      out_samples.reserve(options.samples);
      for (std::size_t s = 0; s < options.samples; ++s)
      {
        Sample sample{};
        double t0 = 0.0;
        if (!NowUsec(t0))
        {
          std::cerr << "RunMicrobenchmark: NowUsec failed" << std::endl;
          return false;
        }
        uint64_t c0 = kHaveRdtsc && options.capture_cycles ? ReadCycles() : 0ull;

        // Run iterations
        try
        {
          for (std::size_t i = 0; i < options.iterations; ++i)
            func(i);
        }
        catch (...)
        {
          std::cerr << "RunMicrobenchmark: exception during measurement" << std::endl;
          return false;
        }

        double t1 = 0.0;
        if (!NowUsec(t1))
        {
          // best-effort: still push zero
          sample.usec = 0.0;
        }
        else
        {
          sample.usec = t1 - t0;
        }
        if (kHaveRdtsc && options.capture_cycles)
        {
          uint64_t c1 = ReadCycles();
          sample.cycles = (c1 >= c0) ? (c1 - c0) : 0ull;
        }
        else
        {
          sample.cycles = 0ull;
        }

        out_samples.push_back(sample);
        // small pause between samples
        std::this_thread::yield();
      }

      return true;
    }

    // ----------------------------- Summarize -----------------------------
    bool Summarize(const std::vector<Sample> &samples, Summary &out_summary) noexcept
    {
      out_summary = Summary{};
      if (samples.empty())
      {
        return false;
      }
      std::size_t n = samples.size();
      out_summary.runs = n;

      std::vector<double> usecs;
      usecs.reserve(n);
      std::vector<double> cyclesd;
      cyclesd.reserve(n);
      std::vector<uint64_t> cyclesu;
      cyclesu.reserve(n);
      bool have_cycles = false;
      for (const auto &s : samples)
      {
        usecs.push_back(s.usec / static_cast<double>(s.usec >= 0.0 ? 1.0 : 1.0)); // retain usec
        cyclesu.push_back(s.cycles);
        cyclesd.push_back(static_cast<double>(s.cycles));
        if (s.cycles != 0ull)
          have_cycles = true;
      }

      out_summary.mean_usec = mean(usecs);
      out_summary.median_usec = median(usecs);
      out_summary.stddev_usec = stddev(usecs, out_summary.mean_usec);
      out_summary.min_usec = *std::min_element(usecs.begin(), usecs.end());
      out_summary.max_usec = *std::max_element(usecs.begin(), usecs.end());

      if (have_cycles)
      {
        out_summary.mean_cycles = mean(cyclesd);
        out_summary.median_cycles = median(cyclesd);
        out_summary.stddev_cycles = stddev(cyclesd, out_summary.mean_cycles);
        out_summary.min_cycles = *std::min_element(cyclesu.begin(), cyclesu.end());
        out_summary.max_cycles = *std::max_element(cyclesu.begin(), cyclesu.end());
      }
      else
      {
        out_summary.mean_cycles = out_summary.median_cycles = out_summary.stddev_cycles = 0.0;
        out_summary.min_cycles = out_summary.max_cycles = 0ull;
      }

      return true;
    }

    std::string SummaryToString(const Summary &s) noexcept
    {
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(3);
      ss << "Runs: " << s.runs << "\n";
      ss << "Usec -> mean: " << s.mean_usec << "  median: " << s.median_usec
         << "  stddev: " << s.stddev_usec << "  min: " << s.min_usec
         << "  max: " << s.max_usec << "\n";
      if (s.mean_cycles > 0.0)
      {
        ss << "Cycles -> mean: " << s.mean_cycles << "  median: " << s.median_cycles
           << "  stddev: " << s.stddev_cycles << "  min: " << s.min_cycles
           << "  max: " << s.max_cycles << "\n";
      }
      else
      {
        ss << "Cycles -> unavailable on this platform or disabled\n";
      }
      return ss.str();
    }

    // ----------------------------- C wrapper -----------------------------
    extern "C"
    {
      int ffm_run_benchmark(bench_cb_t cb, const CRunOptions *opts, Summary *out_summary)
      {
        if (!cb)
          return -1;
        RunOptions ro;
        if (opts)
        {
          ro.warmup_iterations = opts->warmup_iterations;
          ro.iterations = opts->iterations;
          ro.samples = opts->samples;
          ro.capture_cycles = (opts->capture_cycles != 0);
        }
        std::vector<Sample> samples;
        auto wrapper = [&cb](std::size_t i)
        {
          // call user callback - must be noexcept
          cb(i);
        };
        bool ok = RunMicrobenchmark(wrapper, ro, samples);
        if (!ok)
          return -2;
        Summary s;
        if (!Summarize(samples, s))
          return -3;
        if (out_summary)
        {
          try
          {
            *out_summary = s;
          }
          catch (...)
          {
            return -4;
          }
        }
        return 0;
      }
    } // extern "C"

  } // namespace microbench
} // namespace jcore
