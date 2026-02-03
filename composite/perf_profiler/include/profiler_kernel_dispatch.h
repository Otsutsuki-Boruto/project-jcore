#pragma once

#include "profiler_api.h"
#include "k_kernel_dispatch.h"
#include <string>
#include <unordered_map>
#include <mutex>

namespace jcore::profiler
{

  /**
   * Kernel Dispatch Profiling Integration
   *
   * Wraps kernel dispatch calls with profiling to track:
   * - Which kernel was selected
   * - Performance of each kernel variant
   * - Hardware counters per kernel
   */

  struct KernelSelectionInfo
  {
    std::string kernel_name;
    size_t invocations = 0;
    double total_time_usec = 0.0;
    double avg_time_usec = 0.0;
  };

  class KernelDispatchProfiler
  {
  public:
    static KernelDispatchProfiler &Instance();

    /**
     * Initialize kernel dispatch with profiling enabled
     * Must be called after profiler Init()
     */
    bool Init();

    /**
     * Shutdown kernel dispatch profiling
     */
    void Shutdown();

    /**
     * Profiled matmul dispatch
     * Automatically profiles and tracks kernel selection
     */
    int ProfiledMatmul(const float *A, const float *B, float *C,
                       size_t M, size_t N, size_t K);

    /**
     * Get kernel selection statistics
     */
    std::unordered_map<std::string, KernelSelectionInfo> GetKernelStats() const;

    /**
     * Get last selected kernel name
     */
    std::string GetLastSelectedKernel() const;

    /**
     * Export kernel selection info as JSON
     */
    std::string ExportKernelSelectionJSON() const;

  private:
    KernelDispatchProfiler() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, KernelSelectionInfo> kernel_stats_;
    bool initialized_ = false;
  };

  /**
   * Helper function: Initialize both profiler and kernel dispatch
   */
  bool InitProfiledKernelDispatch();

  /**
   * Helper function: Shutdown both systems
   */
  void ShutdownProfiledKernelDispatch();

} // namespace jcore::profiler