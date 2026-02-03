#include "numa_memory_manager.h"
#include <iostream>
#include <mutex>
#include <thread>

#include "cpu_info.h"
#include "jcore_hw_introspect.h"
#include "mem_wrapper.h"
#include "numa_helper.h"
#include "ffm_hugepage.h"
#include "thread_scheduler.h"
#include "config.h"
#include "pool_manager.h"

using namespace jcore::config;

namespace jcore::numa
{
  std::unordered_map<void *, AllocatorType> allocation_map_;
  std::mutex alloc_map_mutex_;

  static std::once_flag g_init_once;
  static NumaMemoryManager *g_mgr = nullptr;

  NumaMemoryManager::NumaMemoryManager() noexcept
      : initialized_(false), default_node_(-1), use_hugepages_(false), max_node_(0), pool_manager_(nullptr) {}

  NumaMemoryManager::~NumaMemoryManager() noexcept { Shutdown(); }

  Result NumaMemoryManager::Init() noexcept
  {
    if (initialized_)
      return {true, "Already initialized"};

    cpu_info_t cpu = detect_cpu_info();
    if (cpu.numa_nodes <= 0)
      return {false, "NUMA topology not detected"};

    max_node_ = cpu.numa_nodes - 1;

    Config cfg;
    cfg.ParseEnv();
    cfg.Validate();

    if (cfg.memory_mode == MemoryMode::HugePages)
    {
      if (ffm_huge_init() == 0)
        use_hugepages_ = true;
      else
        std::cerr << "[WARN] Hugepage init failed; falling back\n";
    }

    if (ffm_init(FFM_BACKEND_AUTO) != FFM_OK)
      return {false, "Allocator init failed"};

    numa_helper_init();

    default_node_ = (cfg.numa_node >= 0) ? cfg.numa_node : 0;
    if (default_node_ > max_node_)
      default_node_ = 0;

    // Initialize memory pool manager
    size_t pool_size = 512 * 1024 * 1024; // 512 MB
    size_t chunk_size = 64 * 1024; // 64 KB chunks
    pm_status_t pm_status = pm_init(&pool_manager_, pool_size, chunk_size, use_hugepages_ ? 1 : 0, default_node_);
    if (pm_status != PM_OK)
    {
      std::cerr << "[WARN] Pool manager init failed, using direct allocation\n";
      pool_manager_ = nullptr;
    }

    initialized_ = true;
    return {true, "Initialized"};
  }

  void *NumaMemoryManager::Allocate(std::size_t size, int node) noexcept
  {
    if (!initialized_ || size == 0)
      return nullptr;

    void *ptr = nullptr;
    AllocatorType used_allocator = AllocatorType::None;

    int target_node = (node >= 0) ? node : default_node_;
    if (target_node > max_node_)
    {
      std::cerr << "[WARN] Requested node " << target_node
                << " exceeds available nodes (max: " << max_node_
                << "). Using node 0.\n";
      target_node = 0;
    }

    // Try pool manager first
    if (pool_manager_)
    {
      ptr = pm_alloc(pool_manager_);
      if (ptr)
      {
        used_allocator = AllocatorType::PoolManager;
      }
    }

    // Hugepage allocation
    if (!ptr && use_hugepages_)
    {
      ffm_huge_region_t *region = ffm_huge_alloc(size, 1);
      if (region)
      {
        ptr = ffm_huge_ptr(region);
        ffm_huge_touch(region, 0xDEADBEEF);
        used_allocator = AllocatorType::FFM;
      }
      else
        use_hugepages_ = false;
    }

    // NUMA allocation
    if (!ptr)
    {
      ptr = numa_helper_alloc_on_node(size, target_node);
      if (ptr)
        used_allocator = AllocatorType::NUMA_HELPER;
    }

    // Generic FFM fallback
    if (!ptr)
    {
      ptr = ffm_aligned_alloc(64, size);
      if (ptr)
        used_allocator = AllocatorType::FFM;
    }

    if (!ptr)
    {
      std::cerr << "[ERROR] Allocation failed (" << size << " bytes)\n";
      return nullptr;
    }

    // Track allocator
    {
      std::lock_guard<std::mutex> lk(alloc_map_mutex_);
      allocation_map_[ptr] = used_allocator;
    }

    return ptr;
  }

  void NumaMemoryManager::Free(void *ptr, std::size_t size) noexcept
  {
    if (!ptr)
      return;

    AllocatorType allocator = AllocatorType::None;
    {
      std::lock_guard<std::mutex> lk(alloc_map_mutex_);
      auto it = allocation_map_.find(ptr);
      if (it != allocation_map_.end())
      {
        allocator = it->second;
        allocation_map_.erase(it);
      }
    }

    switch (allocator)
    {
      case AllocatorType::PoolManager:
        if (pool_manager_)
          pm_free(pool_manager_, ptr);
        break;
      case AllocatorType::NUMA_HELPER:
        numa_helper_free(ptr, size);
        break;
      case AllocatorType::FFM:
        ffm_free(ptr);
        break;
      default:
        std::cerr << "[ERROR] Free called on unknown pointer\n";
        break;
    }
  }

  void NumaMemoryManager::Shutdown() noexcept
  {
    if (!initialized_)
      return;

    if (pool_manager_)
    {
      pm_shutdown(pool_manager_);
      pool_manager_ = nullptr;
    }

    ffm_huge_shutdown();
    ffm_shutdown();
    initialized_ = false;
  }

  int NumaMemoryManager::GetMaxNode() const noexcept
  {
    return max_node_;
  }

  std::string NumaMemoryManager::Info() const noexcept
  {
    cpu_info_t info = detect_cpu_info();
    std::string out = "NUMA Manager Info:\n";
    out += "  NUMA nodes : " + std::to_string(info.numa_nodes) + "\n";
    out += "  Default node : " + std::to_string(default_node_) + "\n";
    out += "  HugePages : " + std::string(use_hugepages_ ? "Enabled" : "Disabled") + "\n";
    return out;
  }
}

// ==================================================================
// FFM-C compatible C API
extern "C"
{
  static std::mutex g_lock;

  int numa_manager_init(void)
  {
    std::lock_guard<std::mutex> lk(g_lock);
    if (jcore::numa::g_mgr)
      return 0;
    jcore::numa::g_mgr = new jcore::numa::NumaMemoryManager();
    auto r = jcore::numa::g_mgr->Init();
    if (!r.ok)
    {
      delete jcore::numa::g_mgr;
      jcore::numa::g_mgr = nullptr;
      std::cerr << "[ERROR] " << r.message << "\n";
      return -1;
    }
    return 0;
  }

  void *numa_manager_alloc(size_t size, int node)
  {
    std::lock_guard<std::mutex> lk(g_lock);
    if (!jcore::numa::g_mgr)
      return nullptr;
    return jcore::numa::g_mgr->Allocate(size, node);
  }

  void numa_manager_free(void *ptr, size_t size)
  {
    std::lock_guard<std::mutex> lk(g_lock);
    if (jcore::numa::g_mgr)
      jcore::numa::g_mgr->Free(ptr, size);
  }

  void numa_manager_shutdown(void)
  {
    std::lock_guard<std::mutex> lk(g_lock);
    if (jcore::numa::g_mgr)
    {
      jcore::numa::g_mgr->Shutdown();
      delete jcore::numa::g_mgr;
      jcore::numa::g_mgr = nullptr;
    }
  }

  int numa_manager_get_max_node(void)
  {
    std::lock_guard<std::mutex> lk(g_lock);
    if (jcore::numa::g_mgr)
      return jcore::numa::g_mgr->GetMaxNode();
    return -1;
  }
}
