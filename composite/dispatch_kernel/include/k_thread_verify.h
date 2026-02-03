#ifndef K_THREADING_VERIFY_H
#define K_THREADING_VERIFY_H

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <atomic>

#ifdef __cplusplus
extern "C"
{
#endif

  // Weak symbols for runtime detection of library threading functions
  // These will resolve if the libraries are linked with threading support

  // OpenBLAS threading detection
  extern int openblas_get_num_threads(void) __attribute__((weak));
  extern void openblas_set_num_threads(int) __attribute__((weak));

// BLIS threading detection (if directly linked)
// Note: BLIS typically uses OpenMP, so we check OpenMP instead
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
}
#endif

namespace threading_verify
{
  // Result structure
  struct ThreadingStatus
  {
    bool openblas_threading_available;
    int openblas_active_threads;
    bool blis_threading_available;
    int blis_active_threads;
    bool omp_available;
    int omp_active_threads;
    int hardware_threads;
    bool threading_likely_active;
  };

  // Verify threading configuration
  inline ThreadingStatus check_threading_status()
  {
    ThreadingStatus status = {};
    status.hardware_threads = std::thread::hardware_concurrency();

    // Check OpenBLAS
    if (openblas_get_num_threads != nullptr)
    {
      status.openblas_threading_available = true;
      status.openblas_active_threads = openblas_get_num_threads();
    }
    else
    {
      status.openblas_threading_available = false;
      status.openblas_active_threads = 0;
    }

    // Check OpenMP (used by BLIS typically)
#ifdef _OPENMP
    status.omp_available = true;
    status.omp_active_threads = omp_get_max_threads();
    status.blis_threading_available = true;
    status.blis_active_threads = status.omp_active_threads;
#else
    status.omp_available = false;
    status.omp_active_threads = 0;
    status.blis_threading_available = false;
    status.blis_active_threads = 0;
#endif

    // Determine if threading is likely active
    status.threading_likely_active =
        (status.openblas_active_threads > 1) ||
        (status.omp_active_threads > 1);

    return status;
  }

  // Print threading status
  inline void print_threading_status()
  {
    ThreadingStatus status = check_threading_status();

    std::cerr << "\n=== Threading Verification Report ===\n";
    std::cerr << "Hardware threads detected: " << status.hardware_threads << "\n\n";

    std::cerr << "OpenBLAS Threading:\n";
    if (status.openblas_threading_available)
    {
      std::cerr << "  Status: AVAILABLE\n";
      std::cerr << "  Active threads: " << status.openblas_active_threads << "\n";
      if (status.openblas_active_threads <= 1)
        std::cerr << "  WARNING: Running single-threaded!\n";
    }
    else
    {
      std::cerr << "  Status: NOT AVAILABLE (not linked or symbols not exposed)\n";
      std::cerr << "  This may mean OpenBLAS is not directly linked.\n";
    }

    std::cerr << "\nBLIS/OpenMP Threading:\n";
    if (status.omp_available)
    {
      std::cerr << "  Status: AVAILABLE\n";
      std::cerr << "  Active threads: " << status.omp_active_threads << "\n";
      if (status.omp_active_threads <= 1)
        std::cerr << "  WARNING: OpenMP running single-threaded!\n";
    }
    else
    {
      std::cerr << "  Status: NOT AVAILABLE (not compiled with OpenMP)\n";
      std::cerr << "  BLIS will run single-threaded.\n";
    }

    std::cerr << "\nOverall Assessment:\n";
    if (status.threading_likely_active)
    {
      std::cerr << "  ✓ Multi-threading appears to be ACTIVE\n";
    }
    else
    {
      std::cerr << "  ✗ Multi-threading appears to be INACTIVE\n";
      std::cerr << "  Possible causes:\n";
      std::cerr << "    - Libraries not compiled with threading support\n";
      std::cerr << "    - Environment variables not set correctly\n";
      std::cerr << "    - Runtime threading disabled\n";
    }
    std::cerr << "=====================================\n\n";
  }

  // Simple CPU burn test to verify parallelism
  inline double measure_parallel_work(int num_threads)
  {
    const size_t work_per_thread = 100000000;
    std::atomic<size_t> counter{0};

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
    {
      threads.emplace_back([&counter, work_per_thread]()
                           {
        for (size_t j = 0; j < work_per_thread; ++j)
        {
          counter.fetch_add(1, std::memory_order_relaxed);
        } });
    }

    for (auto &t : threads)
      t.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    return elapsed.count();
  }

  // Verify system can actually run multiple threads efficiently
  inline void verify_system_threading()
  {
    std::cerr << "\n=== System Threading Verification ===\n";

    double time_1thread = measure_parallel_work(1);
    double time_4threads = measure_parallel_work(4);

    double speedup = time_1thread / time_4threads;

    std::cerr << "Simple parallel test:\n";
    std::cerr << "  1 thread:  " << time_1thread << " seconds\n";
    std::cerr << "  4 threads: " << time_4threads << " seconds\n";
    std::cerr << "  Speedup:   " << speedup << "x\n";

    if (speedup >= 2.5)
    {
      std::cerr << "  ✓ System threading appears functional\n";
    }
    else if (speedup >= 1.5)
    {
      std::cerr << "  ⚠ Threading working but suboptimal (overhead/contention?)\n";
    }
    else
    {
      std::cerr << "  ✗ Threading not providing expected speedup\n";
      std::cerr << "  This could indicate CPU throttling or scheduling issues\n";
    }
    std::cerr << "====================================\n\n";
  }

  // Force threading setup (call this if environment vars aren't working)
  inline void force_threading_setup()
  {
    int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0)
      hw_threads = 4;

    std::cerr << "[FORCE_THREADING] Setting up for " << hw_threads << " threads\n";

    // Try direct API calls if available
    if (openblas_set_num_threads != nullptr)
    {
      openblas_set_num_threads(hw_threads);
      std::cerr << "[FORCE_THREADING] OpenBLAS set via API\n";
    }

#ifdef _OPENMP
    omp_set_num_threads(hw_threads);
    std::cerr << "[FORCE_THREADING] OpenMP set via API\n";
#endif

    // Also set environment variables as backup
    std::string threads_str = std::to_string(hw_threads);
    setenv("OPENBLAS_NUM_THREADS", threads_str.c_str(), 1);
    setenv("BLIS_NUM_THREADS", threads_str.c_str(), 1);
    setenv("OMP_NUM_THREADS", threads_str.c_str(), 1);
    setenv("GOTO_NUM_THREADS", threads_str.c_str(), 1);

    std::cerr << "[FORCE_THREADING] Environment variables set\n";
  }

} // namespace threading_verify

#endif // K_THREADING_VERIFY_H