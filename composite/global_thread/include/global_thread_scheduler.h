#pragma once
#include <cstddef>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <mutex>
#include "thread_scheduler.h"
#include "numa_memory_manager.h"
#include "config.h"
#include "jcore_hw_introspect.h"
#include "cpu_info.h"

typedef struct pm_t pm_t;

namespace jcore::global_thread
{
  struct SchedulerResult
  {
    bool ok;
    std::string message;
    SchedulerResult(bool b = true, std::string m = "") : ok(b), message(std::move(m)) {}
  };

  class GlobalThreadScheduler
  {
  public:
    GlobalThreadScheduler() noexcept;
    ~GlobalThreadScheduler() noexcept;

    // Initialize scheduler with config and NUMA memory manager
    SchedulerResult Init(const jcore::config::Config &cfg) noexcept;

    // Shutdown scheduler
    void Shutdown() noexcept;

    // Submit parallel job
    bool ParallelFor(std::size_t n, const std::function<void(std::size_t)> &worker) noexcept;

    // Query number of threads
    std::size_t GetNumThreads() const noexcept;

    // Query backend name
    std::string BackendName() const noexcept;

    // Info string
    std::string Info() const noexcept;

    // NUMA-aware allocation
    void *AllocateNUMA(std::size_t size, int node = -1) noexcept
    {
      if (!numa_mgr_)
        return nullptr;
      return numa_mgr_->Allocate(size, node);
    }

    // NUMA-aware free
    void FreeNUMA(void *ptr, std::size_t size = 0) noexcept
    {
      if (numa_mgr_)
        numa_mgr_->Free(ptr, size);
    }

    // Expose NUMA info
    std::string NUMAInfo() const noexcept
    {
      return numa_mgr_ ? numa_mgr_->Info() : "N/A";
    }

  private:
    ThreadScheduler scheduler_;
    jcore::numa::NumaMemoryManager *numa_mgr_;
    std::size_t num_threads_;
    bool initialized_;
    mutable std::mutex mutex_;
    pm_t *pool_manager_;

    // Helper: pin threads to NUMA nodes
    void PinThreadsToNUMA() noexcept;

    // Helper: validate CPU features
    bool ValidateCPUFeatures() noexcept;
  };
}
