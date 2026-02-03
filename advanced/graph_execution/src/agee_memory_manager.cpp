// advanced/src/agee_memory_manager.cpp
#include "agee_internal.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

#include "mem_wrapper.h"

namespace agee_internal {

/* ========================================================================== */
/* Memory Usage Prediction                                                    */
/* ========================================================================== */

int PredictMemoryUsage(GraphPlanImpl *plan) {
  if (!plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  // Simple analysis: sum of all tensor sizes
  // In a more sophisticated implementation, we would analyze lifetime
  // and reuse opportunities

  size_t total_memory = 0;
  size_t peak_memory = 0;

  // Track active tensors at each execution step
  std::unordered_map<uint64_t, bool> tensor_active;

  for (const auto &pair : plan->tensors) {
    tensor_active[pair.first] = false;
  }

  // Simulate execution and track peak memory
  for (uint64_t node_id : plan->execution_order) {
    auto node_it = std::find_if(
        plan->execution_nodes.begin(), plan->execution_nodes.end(),
        [node_id](const ExecutionNode &n) { return n.node_id == node_id; });

    if (node_it == plan->execution_nodes.end()) {
      continue;
    }

    // Activate input tensors
    for (uint64_t tid : node_it->input_tensors) {
      tensor_active[tid] = true;
    }

    // Activate output tensors
    for (uint64_t tid : node_it->output_tensors) {
      tensor_active[tid] = true;
    }

    // Calculate current memory usage
    size_t current_memory = 0;
    for (const auto &pair : tensor_active) {
      if (pair.second) {
        auto it = plan->tensors.find(pair.first);
        if (it != plan->tensors.end()) {
          current_memory += it->second.size_bytes;
        }
      }
    }

    peak_memory = std::max(peak_memory, current_memory);

    // Deactivate tensors that are no longer needed
    // (i.e., not used as input by any subsequent node)
    for (uint64_t tid : node_it->output_tensors) {
      bool still_needed = false;

      // Check if any subsequent node needs this tensor
      auto current_pos = std::find(plan->execution_order.begin(),
                                   plan->execution_order.end(), node_id);
      if (current_pos != plan->execution_order.end()) {
        ++current_pos;

        for (auto it = current_pos; it != plan->execution_order.end(); ++it) {
          auto future_node = std::find_if(
              plan->execution_nodes.begin(), plan->execution_nodes.end(),
              [&](const ExecutionNode &n) { return n.node_id == *it; });

          if (future_node != plan->execution_nodes.end()) {
            if (std::find(future_node->input_tensors.begin(),
                          future_node->input_tensors.end(),
                          tid) != future_node->input_tensors.end()) {
              still_needed = true;
              break;
            }
          }
        }
      }

      if (!still_needed) {
        tensor_active[tid] = false;
      }
    }
  }

  plan->peak_memory_bytes = peak_memory;

  // Account for memory saved through fusion
  for (const auto &group : plan->fusion_groups) {
    if (group.estimated_memory_saved > 0) {
      plan->peak_memory_bytes =
          plan->peak_memory_bytes > group.estimated_memory_saved
              ? plan->peak_memory_bytes - group.estimated_memory_saved
              : 0;
    }
  }

  if (g_agee_state.global_config.verbose) {
    printf("[AGEE] Predicted peak memory: %.2f MB\n",
           plan->peak_memory_bytes / (1024.0 * 1024.0));
  }

  return AGEE_OK;
}

/* ========================================================================== */
/* Allocate Pooled Memory                                                     */
/* ========================================================================== */

int AllocatePooledMemory(SessionImpl *session, GraphPlanImpl *plan) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  if (plan->memory_preallocated) {
    return AGEE_OK; // Already allocated
  }

  // Allocate memory for each tensor
  for (auto &pair : plan->tensors) {
    TensorDescriptor &desc = pair.second;

    // Skip if already allocated or constant
    if (desc.data || desc.is_constant) {
      continue;
    }

    // Try pool first if enabled
    bool allocated = false;

    if (session->config.enable_memory_pooling && session->memory_pool) {
      pm_stats_t stats = {};
      if (pm_get_stats(session->memory_pool, &stats) == PM_OK) {
        if (desc.size_bytes <= stats.chunk_bytes) {
          void *ptr = pm_alloc(session->memory_pool);
          if (ptr) {
            desc.data = ptr;
            desc.alloc_source = TensorDescriptor::AllocSource::POOL;
            desc.is_owned = true;
            memset(ptr, 0, desc.size_bytes);
            allocated = true;
          }
        }
      }
    }

    // Fallback to NUMA or standard allocation
    if (!allocated) {
      if (session->config.enable_numa_optimization && desc.numa_node >= 0) {
        desc.data = numa_manager_alloc(desc.size_bytes, desc.numa_node);
        if (desc.data) {
          desc.alloc_source = TensorDescriptor::AllocSource::NUMA;
          desc.is_owned = true;
          memset(desc.data, 0, desc.size_bytes);
          allocated = true;
        }
      }

      if (!allocated) {
        desc.data = ffm_malloc(desc.size_bytes);
        if (desc.data) {
          desc.alloc_source = TensorDescriptor::AllocSource::STANDARD;
          desc.is_owned = true;
          memset(desc.data, 0, desc.size_bytes);
          allocated = true;
        }
      }
    }

    if (!allocated) {
      fprintf(stderr, "[AGEE] Failed to allocate tensor %lu (%zu bytes)\n",
              desc.tensor_id, desc.size_bytes);
      return AGEE_ERR_OOM;
    }

    // Enable prefetch if configured
    if (session->config.enable_prefetch) {
      desc.prefetch_enabled = true;
      desc.prefetch_distance = PREFETCH_DISTANCE;
    }
  }

  plan->memory_preallocated = true;

  if (session->config.verbose) {
    printf("[AGEE] Preallocated memory for %zu tensors\n",
           plan->tensors.size());
  }

  return AGEE_OK;
}

/* ========================================================================== */
/* Free Plan Memory                                                           */
/* ========================================================================== */

int FreePlanMemory(SessionImpl *session, GraphPlanImpl *plan) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  if (!plan->memory_preallocated) {
    return AGEE_OK; // Nothing to free
  }

  // Free all allocated tensors that we own
  for (auto &pair : plan->tensors) {
    TensorDescriptor &desc = pair.second;

    // Only free if we allocated it, we own it, and it's not constant
    if (!desc.data || !desc.is_owned || desc.is_constant) {
      continue;
    }

    // Free based on allocation source
    switch (desc.alloc_source) {
    case TensorDescriptor::AllocSource::POOL:
      if (session->memory_pool) {
        pm_free(session->memory_pool, desc.data);
      }
      break;

    case TensorDescriptor::AllocSource::NUMA:
      numa_manager_free(desc.data, desc.size_bytes);
      break;

    case TensorDescriptor::AllocSource::STANDARD:
      ffm_free(desc.data);
      break;

    default:
      // NONE or unknown - don't free
      break;
    }

    desc.data = nullptr;
    desc.is_owned = false;
    desc.alloc_source = TensorDescriptor::AllocSource::NONE;
  }

  plan->memory_preallocated = false;

  return AGEE_OK;
}

/* ========================================================================== */
/* Allocate Individual Tensor                                                 */
/* ========================================================================== */

void *AllocateTensor(SessionImpl *session, size_t size_bytes, int numa_node) {
  if (!session || size_bytes == 0) {
    return nullptr;
  }

  void *ptr = nullptr;

  // Try memory pool first if enabled and size is small enough
  if (session->config.enable_memory_pooling && session->memory_pool) {
    pm_stats_t stats = {};
    if (pm_get_stats(session->memory_pool, &stats) == PM_OK) {
      if (size_bytes <= stats.chunk_bytes) {
        ptr = pm_alloc(session->memory_pool);
        if (ptr) {
          memset(ptr, 0, size_bytes);
          return ptr; // Successfully allocated from pool
        }
      }
    }
  }

  // Fallback to NUMA-aware or standard allocation
  if (session->config.enable_numa_optimization && numa_node >= 0) {
    ptr = numa_manager_alloc(size_bytes, numa_node);
  } else {
    ptr = ffm_malloc(size_bytes);
  }

  // Initialize to zero
  if (ptr) {
    memset(ptr, 0, size_bytes);
  }

  return ptr;
}

} // namespace agee_internal

using namespace agee_internal;

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

extern "C" {

int agee_estimate_memory(agee_graph_plan_t plan, size_t *out_peak_bytes) {
  if (!plan || !out_peak_bytes) {
    return AGEE_ERR_INVALID_ARG;
  }

  GraphPlanImpl *impl = reinterpret_cast<GraphPlanImpl *>(plan);

  // Run prediction if not already done
  if (impl->peak_memory_bytes == 0) {
    int ret = PredictMemoryUsage(impl);
    if (ret != AGEE_OK) {
      return ret;
    }
  }

  *out_peak_bytes = impl->peak_memory_bytes;
  return AGEE_OK;
}

int agee_preallocate_memory(agee_session_t session, agee_graph_plan_t plan) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  if (!g_agee_state.initialized) {
    return AGEE_ERR_NOT_INITIALIZED;
  }

  SessionImpl *sess_impl = reinterpret_cast<SessionImpl *>(session);
  GraphPlanImpl *plan_impl = reinterpret_cast<GraphPlanImpl *>(plan);

  return AllocatePooledMemory(sess_impl, plan_impl);
}

int agee_export_plan_dot(agee_graph_plan_t plan, const char *filename) {
  if (!plan || !filename) {
    return AGEE_ERR_INVALID_ARG;
  }

  GraphPlanImpl *impl = reinterpret_cast<GraphPlanImpl *>(plan);

  // Use operator graph's export function
  if (impl->original_graph) {
    return og_export_graph_dot(impl->original_graph, filename);
  }

  return AGEE_ERR_INTERNAL;
}

int agee_self_test(int verbose) {
  if (!g_agee_state.initialized) {
    fprintf(stderr, "[AGEE] Error: Not initialized\n");
    return AGEE_ERR_NOT_INITIALIZED;
  }

  if (verbose) {
    printf("[AGEE] Running self-test...\n");
  }

  // Test 1: Create and destroy session
  agee_session_t session = nullptr;
  int ret = agee_create_session(&session);
  if (ret != AGEE_OK) {
    fprintf(stderr, "[AGEE] Test failed: Session creation\n");
    return ret;
  }

  if (verbose) {
    printf("[AGEE] ✓ Session creation\n");
  }

  // Test 2: Get system info
  char *info = nullptr;
  ret = agee_get_system_info(&info);
  if (ret != AGEE_OK) {
    fprintf(stderr, "[AGEE] Test failed: System info\n");
    agee_destroy_session(session);
    return ret;
  }

  if (verbose) {
    printf("[AGEE] ✓ System info retrieval\n");
    printf("%s\n", info);
  }

  agee_free_string(info);

  // Test 3: Reset session
  ret = agee_reset_session(session);
  if (ret != AGEE_OK) {
    fprintf(stderr, "[AGEE] Test failed: Session reset\n");
    agee_destroy_session(session);
    return ret;
  }

  if (verbose) {
    printf("[AGEE] ✓ Session reset\n");
  }

  // Cleanup
  agee_destroy_session(session);

  if (verbose) {
    printf("[AGEE] ✓ All tests passed\n");
  }

  return AGEE_OK;
}

int agee_benchmark_plan(agee_session_t session, agee_graph_plan_t plan,
                        int iterations) {
  if (!session || !plan || iterations <= 0) {
    return AGEE_ERR_INVALID_ARG;
  }

  SessionImpl *sess_impl = reinterpret_cast<SessionImpl *>(session);
  GraphPlanImpl *plan_impl = reinterpret_cast<GraphPlanImpl *>(plan);

  if (sess_impl->config.verbose) {
    printf("[AGEE] Benchmarking plan for %d iterations...\n", iterations);
  }

  std::vector<double> execution_times;
  execution_times.reserve(iterations);

  // Preallocate memory once
  int ret = AllocatePooledMemory(sess_impl, plan_impl);
  if (ret != AGEE_OK) {
    fprintf(stderr, "[AGEE] Failed to preallocate memory for benchmark\n");
    return ret;
  }

  for (int i = 0; i < iterations; ++i) {
    agee_exec_stats_t stats = {};
    ret = ExecutePlanInternal(sess_impl, plan_impl, &stats);

    if (ret != AGEE_OK) {
      fprintf(stderr, "[AGEE] Benchmark failed at iteration %d\n", i);
      FreePlanMemory(sess_impl, plan_impl);
      return ret;
    }

    execution_times.push_back(stats.total_execution_time_ms);
  }

  // Free memory after benchmark
  FreePlanMemory(sess_impl, plan_impl);

  // Compute statistics
  if (execution_times.empty()) {
    return AGEE_ERR_INTERNAL;
  }

  std::sort(execution_times.begin(), execution_times.end());

  double min_time = execution_times.front();
  double max_time = execution_times.back();
  double median_time = execution_times[execution_times.size() / 2];

  double sum = 0.0;
  for (double t : execution_times) {
    sum += t;
  }
  double mean_time = sum / execution_times.size();

  // Compute standard deviation
  double variance = 0.0;
  for (double t : execution_times) {
    variance += (t - mean_time) * (t - mean_time);
  }
  double stddev = std::sqrt(variance / execution_times.size());

  if (sess_impl->config.verbose) {
    printf("[AGEE] Benchmark Results (%d iterations):\n", iterations);
    printf("  Min:    %.3f ms\n", min_time);
    printf("  Max:    %.3f ms\n", max_time);
    printf("  Mean:   %.3f ms\n", mean_time);
    printf("  Median: %.3f ms\n", median_time);
    printf("  Stddev: %.3f ms\n", stddev);
  }

  return AGEE_OK;
}

} // extern "C"