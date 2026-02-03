#pragma once
#include "adaptive_tuner.h"
#include "jcore_isa_dispatch.h"
#include <thread>
#include <cstdlib> // setenv
#include <sstream>
#include "ffm_cache_block.h"  // base component - cache blocking utility
#include "thread_scheduler.h" // base component - scheduler helper (we use only detection/hints)
#include "ffm_prefetch.h"
/**
 * Kernel Dispatch Table / Runtime Selector (Derived Component)
 *
 * Purpose:
 * - Dynamically selects optimal kernel per operation (matmul/vector ops) based on CPU features.
 * - Uses Adaptive Kernel Auto-Tuner and base ISA dispatch features.
 * - Exposes runtime dispatch and initialization.
 */

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * Initialize derived kernel dispatch (calls base + tuner init)
   */
  int k_dispatch_init(void);

  /**
   * Shutdown derived kernel dispatch (cleanup tuner state)
   */
  void k_dispatch_shutdown(void);

  /**
   * Dispatch matmul via derived dispatch (chooses best kernel)
   * Returns JCORE_OK or JCORE_ERR_*
   */
  int k_dispatch_matmul(const float *A, const float *B, float *C,
                        size_t M, size_t N, size_t K);

  const char *k_dispatch_get_last_selected_kernel();

#ifdef __cplusplus
}
#endif
