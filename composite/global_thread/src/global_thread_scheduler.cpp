#include "jcore_isa_dispatch.h"
#include "global_thread_scheduler.h"
#include "ffm_cache_block.h" // <-- added for cache tiling
#include "ffm_prefetch.h"
#include "mem_wrapper.h"
#include "numa_helper.h"
#include "pool_manager.h"

#include <iostream>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace jcore::global_thread
{
  GlobalThreadScheduler::GlobalThreadScheduler() noexcept
      : numa_mgr_(nullptr), num_threads_(0), initialized_(false), pool_manager_(nullptr)
  {
  }

  // Destructor
  GlobalThreadScheduler::~GlobalThreadScheduler() noexcept
  {
    Shutdown();
  }

  // Initialize the scheduler
  SchedulerResult GlobalThreadScheduler::Init(const jcore::config::Config &cfg) noexcept
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_)
      return SchedulerResult(true, "Already initialized");

    // --- ISA Dispatch Initialization ---
    if (jcore_init_dispatch() != JCORE_OK)
      return SchedulerResult(false, "Failed to initialize ISA dispatch");

    jcore_features_t feats = jcore_get_host_features();
    if (!(feats & (JCORE_FEAT_AVX | JCORE_FEAT_AVX2 | JCORE_FEAT_AVX512)))
      return SchedulerResult(false, "CPU lacks minimum AVX feature set");

    // --- Initialize NUMA Manager ---
    numa_mgr_ = new jcore::numa::NumaMemoryManager();

    // ------------------------------------------------------------
    // Initialize memory allocator wrapper
    // ------------------------------------------------------------
    if (ffm_init(FFM_BACKEND_AUTO) != FFM_OK)
    {
      return {false, "Memory allocator initialization failed"};
    }

    // Initialize NUMA helper for allocator integration
    if (numa_helper_init() != 0)
    {
      std::cerr << "[WARN] NUMA helper not available, continuing without NUMA-specific allocation\n";
    }

    // Initialize memory pool manager
    size_t pool_size = 256 * 1024 * 1024; // 256 MB
    size_t chunk_size = 64 * 1024; // 64 KB chunks
    int use_hp = (cfg.memory_mode == jcore::config::MemoryMode::HugePages) ? 1 : 0;
    pm_status_t pm_status = pm_init(&pool_manager_, pool_size, chunk_size, use_hp, cfg.numa_node);
    if (pm_status != PM_OK)
    {
      std::cerr << "[WARN] Pool manager init failed, using FFM allocator\n";
      pool_manager_ = nullptr;
    }

    // Set allocator NUMA node preference from config
    if (cfg.numa_node >= 0 && numa_mgr_ && cfg.numa_node <= numa_mgr_->GetMaxNode())
    {
      ffm_status_t s = ffm_set_numa_node(cfg.numa_node);
      if (s != FFM_OK)
        std::cerr << "[WARN] Failed to set NUMA node in allocator: "
                  << ffm_status_str(s) << std::endl;
    }

    // else
    // {
    //   std::cerr << "[INFO] NUMA node binding skipped (single-node system)\n";
    // }
    //
    // std::cout << "[INFO] Memory allocator backend: " << ffm_get_backend() << std::endl;

    auto numa_res = numa_mgr_->Init();
    if (!numa_res.ok)
      return SchedulerResult(false, "NUMA Manager init failed: " + numa_res.message);

    // --- Initialize Thread Scheduler ---
    auto sched_res = scheduler_.Init(jcore::SchedulerBackend::Auto, cfg.threads);
    if (!sched_res.ok)
      return SchedulerResult(false, "ThreadScheduler init failed: " + sched_res.message);

    num_threads_ = scheduler_.GetNumThreads();

    // --- Optional NUMA Pinning ---
    if (cfg.numa_node >= 0)
      PinThreadsToNUMA();

    // --- Light Cache-Aware Setup ---
    ffm_cache_info_t *cache_info = ffm_cache_init();
    if (cache_info)
    {
      size_t tile = ffm_cache_compute_tile(cache_info, 1, sizeof(double), 0.6); // Light L1-aware tile
      ffm_prefetch_addr_read(cache_info);                                       // Prefetch cache info block to keep metadata hot

      if (tile > 0)
        std::cout << "[INFO] L1 Cache Tile Size: " << tile << " elements per dimension" << std::endl;
      else
        std::cout << "[WARN] Could not compute cache tile; using default partitioning." << std::endl;
      ffm_cache_free(cache_info);
    }
    else
    {
      std::cout << "[WARN] Cache info unavailable; proceeding without tiling." << std::endl;
    }

    initialized_ = true;
    return SchedulerResult(true, "GlobalThreadScheduler initialized (ISA + NUMA + cache tiling ready)");
  }

  void GlobalThreadScheduler::Shutdown() noexcept
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_)
      return;

    scheduler_.Shutdown();

    if (numa_mgr_)
    {
      numa_mgr_->Shutdown();
      delete numa_mgr_;
      numa_mgr_ = nullptr;
    }

    if (pool_manager_)
    {
      pm_shutdown(pool_manager_);
      pool_manager_ = nullptr;
    }

    initialized_ = false;
    // Shutdown allocator before exiting
    ffm_shutdown();
  }

  bool GlobalThreadScheduler::ParallelFor(std::size_t n, const std::function<void(std::size_t)> &worker) noexcept
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !worker)
      return false;

    return scheduler_.ParallelFor(n, worker);
  }

  std::size_t GlobalThreadScheduler::GetNumThreads() const noexcept
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_threads_;
  }

  std::string GlobalThreadScheduler::BackendName() const noexcept
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return scheduler_.BackendName();
  }

  std::string GlobalThreadScheduler::Info() const noexcept
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "GlobalThreadScheduler: " << num_threads_
        << " threads, backend=" << scheduler_.BackendName()
        << ", NUMA info=" << NUMAInfo();
    return oss.str();
  }

  void GlobalThreadScheduler::PinThreadsToNUMA() noexcept
  {
    // lightweight placeholder
  }

  bool GlobalThreadScheduler::ValidateCPUFeatures() noexcept
  {
    jcore_features_t feats = jcore_get_host_features();
    return (feats & (JCORE_FEAT_AVX | JCORE_FEAT_AVX2 | JCORE_FEAT_AVX512));
  }

} // namespace jcore::global_thread
