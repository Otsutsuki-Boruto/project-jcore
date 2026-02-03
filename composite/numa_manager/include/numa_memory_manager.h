#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <mutex>

typedef struct pm_t pm_t;

namespace jcore::numa
{
  enum class AllocatorType
  {
    None,
    NUMA_HELPER,
    PoolManager,
    FFM
  };

  extern std::unordered_map<void *, AllocatorType> allocation_map_;
  extern std::mutex alloc_map_mutex_;
}

namespace jcore::numa
{

  struct Result
  {
    bool ok;
    std::string message;
    Result(bool s = true, std::string m = "") noexcept : ok(s), message(std::move(m)) {}
  };

  class NumaMemoryManager
  {
  public:
    NumaMemoryManager() noexcept;
    ~NumaMemoryManager() noexcept;

    Result Init() noexcept;
    void Shutdown() noexcept;
    int GetMaxNode() const noexcept;

    void *Allocate(std::size_t size, int node = -1) noexcept;
    void Free(void *ptr, std::size_t size = 0) noexcept;
    std::string Info() const noexcept;

  private:
    bool initialized_;
    int default_node_;
    bool use_hugepages_;
    int max_node_; // maximum available NUMA node
    pm_t *pool_manager_;
  };
}

// FFM-compatible C API
extern "C"
{
  int numa_manager_init(void);
  void *numa_manager_alloc(size_t size, int node);
  void numa_manager_free(void *ptr, size_t size);
  void numa_manager_shutdown(void);
  int numa_manager_get_max_node(void);
}
