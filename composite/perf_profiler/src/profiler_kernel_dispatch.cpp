#include "profiler_kernel_dispatch.h"
#include "profiler_region.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace jcore::profiler
{

  KernelDispatchProfiler &KernelDispatchProfiler::Instance()
  {
    static KernelDispatchProfiler instance;
    return instance;
  }

  bool KernelDispatchProfiler::Init()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_)
    {
      return true;
    }

    // Initialize kernel dispatch
    int result = k_dispatch_init();
    if (result != 0)
    {
      std::cerr << "[KernelDispatchProfiler] Failed to initialize kernel dispatch: " << result << "\n";
      return false;
    }

    std::cout << "[KernelDispatchProfiler] Initialized successfully\n";
    initialized_ = true;
    return true;
  }

  void KernelDispatchProfiler::Shutdown()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_)
    {
      return;
    }

    k_dispatch_shutdown();
    kernel_stats_.clear();
    initialized_ = false;

    std::cout << "[KernelDispatchProfiler] Shutdown complete\n";
  }

  int KernelDispatchProfiler::ProfiledMatmul(const float *A, const float *B, float *C,
                                             size_t M, size_t N, size_t K)
  {
    if (!initialized_)
    {
      std::cerr << "[KernelDispatchProfiler] Not initialized!\n";
      return -1;
    }

    // Create unique region name based on operation and dimensions
    std::ostringstream region_name;
    region_name << "matmul_" << M << "x" << N << "x" << K;

    int result = -1;

    // Profile the dispatched matmul
    ProfileRegion(region_name.str(), [&]()
                  { result = k_dispatch_matmul(A, B, C, M, N, K); });

    // Get which kernel was selected
    const char *kernel_name = k_dispatch_get_last_selected_kernel();
    std::string kernel_str = kernel_name ? kernel_name : "unknown";

    // Get the profiling stats for this region
    auto snapshot = RegionRegistry::Instance().Snapshot();
    auto it = snapshot.find(region_name.str());

    if (it != snapshot.end())
    {
      const auto &stats = it->second;

      // Update kernel selection stats
      std::lock_guard<std::mutex> lock(mutex_);
      auto &kernel_info = kernel_stats_[kernel_str];

      kernel_info.kernel_name = kernel_str;
      kernel_info.invocations++;
      kernel_info.total_time_usec += stats.total_usec;
      kernel_info.avg_time_usec = kernel_info.total_time_usec / kernel_info.invocations;
    }

    return result;
  }

  std::unordered_map<std::string, KernelSelectionInfo>
  KernelDispatchProfiler::GetKernelStats() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernel_stats_;
  }

  std::string KernelDispatchProfiler::GetLastSelectedKernel() const
  {
    const char *kernel_name = k_dispatch_get_last_selected_kernel();
    return kernel_name ? kernel_name : "unknown";
  }

  std::string KernelDispatchProfiler::ExportKernelSelectionJSON() const
  {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;
    oss << "{ \"kernel_selections\": [";

    bool first = true;
    for (const auto &kv : kernel_stats_)
    {
      if (!first)
        oss << ",";
      first = false;

      const auto &info = kv.second;
      oss << "{"
          << "\"kernel\":\"" << info.kernel_name << "\","
          << "\"invocations\":" << info.invocations << ","
          << "\"total_time_usec\":" << std::fixed << std::setprecision(2) << info.total_time_usec << ","
          << "\"avg_time_usec\":" << std::fixed << std::setprecision(2) << info.avg_time_usec
          << "}";
    }

    oss << "] }";
    return oss.str();
  }

  // Helper functions

  bool InitProfiledKernelDispatch()
  {
    // Initialize profiler first
    if (!Init())
    {
      std::cerr << "[InitProfiledKernelDispatch] Profiler initialization failed\n";
      return false;
    }

    // Then initialize kernel dispatch profiler
    if (!KernelDispatchProfiler::Instance().Init())
    {
      std::cerr << "[InitProfiledKernelDispatch] Kernel dispatch initialization failed\n";
      Shutdown();
      return false;
    }

    std::cout << "[InitProfiledKernelDispatch] Full system initialized\n";
    return true;
  }

  void ShutdownProfiledKernelDispatch()
  {
    KernelDispatchProfiler::Instance().Shutdown();
    Shutdown();
    std::cout << "[ShutdownProfiledKernelDispatch] Full system shutdown complete\n";
  }

} // namespace jcore::profiler