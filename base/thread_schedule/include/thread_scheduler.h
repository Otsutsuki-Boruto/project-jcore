// include/thread_scheduler.h
#ifndef JC_CORE_THREAD_SCHEDULER_H_
#define JC_CORE_THREAD_SCHEDULER_H_

// Thread Scheduler - Unified control over OpenMP or oneTBB threads
//
// API: FFM-compatible (lightweight) scheduler abstraction used by Project JCore.
// - Supports backends: OpenMP, oneTBB, Serial fallback.
// - Primary responsibilities:
//   * initialize/shutdown scheduler
//   * query/set thread count
//   * parallel_for(range, worker) to run index-based jobs
//
// Design notes:
// - Header is minimal, exception-free by default; most methods return bool for status.
// - The user may provide a lambda: void(size_t idx).
// - Thread count 0 means "auto (hardware_concurrency)".

#include <cstddef>
#include <functional>
#include <string>

namespace jcore
{

  // Forward declare error type (simple struct to carry messages)
  struct SchedulerError
  {
    bool ok;
    std::string message;
    SchedulerError(bool b = true, std::string m = "") : ok(b), message(std::move(m)) {}
  };

  // Which backend to prefer
  enum class SchedulerBackend
  {
    Auto, // pick best available (TBB > OpenMP > Serial)
    TBB,
    OpenMP,
    Serial
  };

  // Small, self-contained scheduler class.
  // Thread-safe usage note: this class is not internally synchronized. The caller should
  // initialize once (single-threaded) then use parallel_for concurrently as intended by the backend.
  class ThreadScheduler
  {
  public:
    // Create scheduler object (lightweight). Nothing heavy in ctor.
    ThreadScheduler() noexcept;

    // Destructor: will call Shutdown() if needed.
    ~ThreadScheduler() noexcept;

    // Initialize scheduler. `requested_threads` of 0 uses hardware_concurrency/auto.
    // Returns SchedulerError.ok==true on success, false otherwise with a descriptive message.
    // If backend==Auto, scheduler will pick the most capable backend compiled-in.
    SchedulerError Init(SchedulerBackend backend = SchedulerBackend::Auto,
                        std::size_t requested_threads = 0) noexcept;

    // Set maximum number of worker threads at runtime (best-effort).
    // Returns true on success; false if unsupported by selected backend.
    bool SetNumThreads(std::size_t num_threads) noexcept;

    // Get current configured number of worker threads (0 meaning single-threaded/serial fallback).
    std::size_t GetNumThreads() const noexcept;

    // Execute worker(index) for index in [0, n). Worker must be safe for parallel invocation.
    // Returns true if executed; false if invalid args or scheduler not initialized.
    bool ParallelFor(std::size_t n, const std::function<void(std::size_t)> &worker) const noexcept;

    // Shutdown/cleanup resources. Safe to call multiple times.
    void Shutdown() noexcept;

    // Retrieve a human readable string of chosen backend
    std::string BackendName() const noexcept;

  private:
    // Non-copyable
    ThreadScheduler(const ThreadScheduler &) = delete;
    ThreadScheduler &operator=(const ThreadScheduler &) = delete;

    // PIMPL-like minimal internal state
    enum class ImplBackend
    {
      None,
      UseTBB,
      UseOpenMP,
      UseSerial
    };
    ImplBackend impl_backend_;
    std::size_t num_threads_; // 0 => serial / single-threaded
    bool initialized_;
  };

} // namespace jcore

#endif // JC_CORE_THREAD_SCHEDULER_H_
