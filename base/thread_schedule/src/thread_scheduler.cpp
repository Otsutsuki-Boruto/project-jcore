// src/thread_scheduler.cpp
#include "thread_scheduler.h"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <thread>

#ifdef __has_include
#if __has_include(<tbb/global_control.h>) && __has_include(<tbb/parallel_for.h>)
#define JCORE_HAVE_TBB 1
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif
#endif

#ifdef _OPENMP
#define JCORE_HAVE_OPENMP 1
#include <omp.h>
#endif

namespace jcore
{

  ThreadScheduler::ThreadScheduler() noexcept
      : impl_backend_(ImplBackend::None), num_threads_(0), initialized_(false) {}

  ThreadScheduler::~ThreadScheduler() noexcept
  {
    Shutdown();
  }

  static std::size_t HardwareConcurrencyOrDefault() noexcept
  {
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0u)
    {
      return 1u;
    }
    return static_cast<std::size_t>(hc);
  }

  SchedulerError ThreadScheduler::Init(SchedulerBackend backend, std::size_t requested_threads) noexcept
  {
    if (initialized_)
    {
      // Re-init allowed: perform Shutdown then re-init
      Shutdown();
    }

    std::size_t hw = HardwareConcurrencyOrDefault();
    std::size_t threads = (requested_threads == 0 ? hw : requested_threads);

    // Decide backend based on requested and compile-time availability
    ImplBackend chosen = ImplBackend::UseSerial;

    if (backend == SchedulerBackend::Auto)
    {
#if defined(JCORE_HAVE_TBB)
      chosen = ImplBackend::UseTBB;
#elif defined(JCORE_HAVE_OPENMP)
      chosen = ImplBackend::UseOpenMP;
#else
      chosen = ImplBackend::UseSerial;
#endif
    }
    else
    {
      switch (backend)
      {
      case SchedulerBackend::TBB:
#if defined(JCORE_HAVE_TBB)
        chosen = ImplBackend::UseTBB;
#else
        return SchedulerError(false, "TBB backend requested but library not available at compile time.");
#endif
        break;
      case SchedulerBackend::OpenMP:
#if defined(JCORE_HAVE_OPENMP)
        chosen = ImplBackend::UseOpenMP;
#else
        return SchedulerError(false, "OpenMP backend requested but compiler not built with OpenMP support.");
#endif
        break;
      case SchedulerBackend::Serial:
        chosen = ImplBackend::UseSerial;
        break;
      default:
        return SchedulerError(false, "Unknown scheduler backend requested.");
      }
    }

    // Apply chosen backend
    impl_backend_ = chosen;
    num_threads_ = threads;
    initialized_ = true;

    // Backend-specific setup
    switch (impl_backend_)
    {
#if defined(JCORE_HAVE_TBB)
    case ImplBackend::UseTBB:
    {
      try
      {
        // set global control for max threads
        // tbb::global_control expects an enum value
        // allocate control on heap but keep it via static-like storage? Instead manage by setting and letting destructor run on Shutdown.
        // We'll use a thread-local/global control object via new and store pointer in a lambda-captured static to avoid exposing in header.
        static tbb::global_control *gcontrol = nullptr;
        // delete previous if exists
        if (gcontrol)
        {
          delete gcontrol;
          gcontrol = nullptr;
        }
        gcontrol = new tbb::global_control(tbb::global_control::max_allowed_parallelism, static_cast<int>(threads));
        (void)gcontrol; // silence unused in some build configs
      }
      catch (const std::exception &e)
      {
        initialized_ = false;
        impl_backend_ = ImplBackend::None;
        return SchedulerError(false, std::string("TBB global_control failed: ") + e.what());
      }
      break;
    }
#endif

#if defined(JCORE_HAVE_OPENMP)
    case ImplBackend::UseOpenMP:
    {
      // Best-effort: omp_set_num_threads is not an error; some runtimes ignore.
      omp_set_num_threads(static_cast<int>(threads));
      break;
    }
#endif

    case ImplBackend::UseSerial:
      // No setup required
      break;

    default:
      initialized_ = false;
      impl_backend_ = ImplBackend::None;
      return SchedulerError(false, "Impossible backend selection.");
    }

    // Success
    return SchedulerError(true, "");
  }

  bool ThreadScheduler::SetNumThreads(std::size_t num_threads) noexcept
  {
    if (!initialized_)
    {
      std::cerr << "ThreadScheduler::SetNumThreads called before Init()" << std::endl;
      return false;
    }
    if (num_threads == 0)
    {
      num_threads = HardwareConcurrencyOrDefault();
    }

    switch (impl_backend_)
    {
#if defined(JCORE_HAVE_TBB)
    case ImplBackend::UseTBB:
    {
      // update tbb global_control (best-effort)
      try
      {
        // Create a new global_control and replace previous one by leaking old pointer? We kept pointer in Init's static var.
        // To avoid raw pointer juggling, we create a local control for duration of call by using existing approach:
        // The simplest portable approach: allocate and delete previous global_control.
        // Unfortunately there is no official API to change it, so we recreate.
        extern tbb::global_control *__jcore_tbb_global_control__;
        (void)__jcore_tbb_global_control__; // will be weak symbol; but to be safe, fallback to new creation below.
      }
      catch (...)
      {
        // ignore
      }
      // Best-effort: create new control (may not replace existing effect depending on tbb version)
      try
      {
        // Try to set through a new control and delete old one if any.
        // Note: we used a static pointer in Init; we will replicate same static here to find and replace.
        static tbb::global_control *gcontrol = nullptr;
        if (gcontrol)
        {
          delete gcontrol;
          gcontrol = nullptr;
        }
        gcontrol = new tbb::global_control(tbb::global_control::max_allowed_parallelism, static_cast<int>(num_threads));
      }
      catch (...)
      {
        return false;
      }
      num_threads_ = num_threads;
      return true;
    }
#endif

#if defined(JCORE_HAVE_OPENMP)
    case ImplBackend::UseOpenMP:
    {
      omp_set_num_threads(static_cast<int>(num_threads));
      num_threads_ = num_threads;
      return true;
    }
#endif

    case ImplBackend::UseSerial:
    {
      // serial backend ignores requests >1
      num_threads_ = 1;
      return true;
    }

    default:
      return false;
    }
  }

  std::size_t ThreadScheduler::GetNumThreads() const noexcept
  {
    if (!initialized_)
    {
      return 1u;
    }
    switch (impl_backend_)
    {
#if defined(JCORE_HAVE_TBB)
    case ImplBackend::UseTBB:
    {
      // tbb does not expose direct current parallelism easily; return configured value
      return num_threads_;
    }
#endif

#if defined(JCORE_HAVE_OPENMP)
    case ImplBackend::UseOpenMP:
    {
      int t = omp_get_max_threads();
      return static_cast<std::size_t>(t > 0 ? t : 1);
    }
#endif

    case ImplBackend::UseSerial:
    default:
      return num_threads_ == 0 ? 1u : num_threads_;
    }
  }

  bool ThreadScheduler::ParallelFor(std::size_t n, const std::function<void(std::size_t)> &worker) const noexcept
  {
    if (!initialized_)
    {
      std::cerr << "ThreadScheduler::ParallelFor called before Init()." << std::endl;
      return false;
    }
    if (!worker)
    {
      std::cerr << "ThreadScheduler::ParallelFor invalid worker function." << std::endl;
      return false;
    }
    if (n == 0)
    {
      return true; // nothing to do
    }

    switch (impl_backend_)
    {
#if defined(JCORE_HAVE_TBB)
    case ImplBackend::UseTBB:
    {
      try
      {
        // Use tbb::parallel_for with blocked_range
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n),
                          [&worker](const tbb::blocked_range<std::size_t> &r)
                          {
                            for (std::size_t i = r.begin(); i != r.end(); ++i)
                            {
                              worker(i);
                            }
                          });
      }
      catch (const std::exception &e)
      {
        std::cerr << "TBB parallel_for exception: " << e.what() << std::endl;
        return false;
      }
      catch (...)
      {
        std::cerr << "TBB parallel_for unknown exception." << std::endl;
        return false;
      }
      return true;
    }
#endif

#if defined(JCORE_HAVE_OPENMP)
    case ImplBackend::UseOpenMP:
    {
      // Use OpenMP for-loop. Be careful to avoid iterator reordering issues.
      // We rely on default(static) schedule here; user may override by setting OMP_SCHEDULE env var if desired.
#pragma omp parallel for schedule(static)
      for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
      {
        try
        {
          worker(static_cast<std::size_t>(i));
        }
        catch (...)
        {
          // Swallow exceptions inside threads; report by writing to stderr.
          // Throwing out of OpenMP parallel region is undefined behavior.
          std::cerr << "Exception in OpenMP worker for index " << i << std::endl;
        }
      }
      return true;
    }
#endif

    case ImplBackend::UseSerial:
    {
      for (std::size_t i = 0; i < n; ++i)
      {
        worker(i);
      }
      return true;
    }

    default:
      std::cerr << "ThreadScheduler::ParallelFor unknown backend." << std::endl;
      return false;
    }
  }

  void ThreadScheduler::Shutdown() noexcept
  {
    if (!initialized_)
      return;

    // Backend-specific tear-down
    switch (impl_backend_)
    {
#if defined(JCORE_HAVE_TBB)
    case ImplBackend::UseTBB:
    {
      // delete global_control if we stored one
      static tbb::global_control *gcontrol = nullptr;
      if (gcontrol)
      {
        delete gcontrol;
        gcontrol = nullptr;
      }
      break;
    }
#endif

    case ImplBackend::UseOpenMP:
    case ImplBackend::UseSerial:
    default:
      break;
    }

    impl_backend_ = ImplBackend::None;
    num_threads_ = 0;
    initialized_ = false;
  }

  std::string ThreadScheduler::BackendName() const noexcept
  {
    switch (impl_backend_)
    {
    case ImplBackend::UseTBB:
      return "oneTBB";
    case ImplBackend::UseOpenMP:
      return "OpenMP";
    case ImplBackend::UseSerial:
      return "Serial";
    default:
      return "None";
    }
  }

} // namespace jcore
