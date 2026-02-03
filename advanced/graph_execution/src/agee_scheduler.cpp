// advanced/src/agee_scheduler.cpp
#include "agee_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace agee_internal {

/* ========================================================================== */
/* Predictive Scheduling - Assign NUMA nodes and thread affinity            */
/* ========================================================================== */

int ScheduleOperations(SessionImpl *session, GraphPlanImpl *plan) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  // Assign NUMA nodes first
  if (session->config.enable_numa_optimization && session->max_numa_nodes > 1) {
    int ret = AssignNUMANodes(session, plan);
    if (ret != AGEE_OK && session->config.verbose) {
      fprintf(stderr, "[AGEE] Warning: NUMA assignment failed\n");
    }
  }

  // Assign thread affinity
  int ret = AssignThreadAffinity(session, plan);
  if (ret != AGEE_OK && session->config.verbose) {
    fprintf(stderr, "[AGEE] Warning: Thread affinity assignment failed\n");
  }

  // Estimate execution times
  for (auto &node : plan->execution_nodes) {
    if (node.estimated_flops > 0) {
      // Estimate based on FLOPS and assumed 100 GFLOPS/s per core
      double cores_used = session->config.num_threads > 0
                              ? session->config.num_threads
                              : session->cpu_info.cores;
      double gflops_per_sec = cores_used * 100.0; // Conservative estimate
      node.estimated_time_ms =
          (node.estimated_flops / 1e9) / gflops_per_sec * 1000.0;
    } else {
      // Assume 0.1ms for non-compute ops
      node.estimated_time_ms = 0.1;
    }

    plan->estimated_total_time_ms += node.estimated_time_ms;
  }

  // Select kernels using cached auto-tuner
  if (session->config.enable_adaptive_tuning && session->cached_tuner) {
    for (auto &node : plan->execution_nodes) {
      if (IsGEMMOperation(node.op_type) && node.input_tensors.size() >= 2) {
        // Get tensor shapes
        auto it1 = plan->tensors.find(node.input_tensors[0]);
        auto it2 = plan->tensors.find(node.input_tensors[1]);

        if (it1 != plan->tensors.end() && it2 != plan->tensors.end() &&
            it1->second.ndim >= 2 && it2->second.ndim >= 2) {

          size_t M = it1->second.shape[0];
          size_t K = it1->second.shape[1];
          size_t N = it2->second.shape[1];

          size_t threads = session->config.num_threads > 0
                               ? session->config.num_threads
                               : session->cpu_info.cores;

          // Get optimal tile size
          size_t tile_size = 64; // Default
          if (g_agee_state.cache_info) {
            tile_size = ffm_cache_compute_tile(g_agee_state.cache_info, 2,
                                               sizeof(float), 0.75);
            if (tile_size == 0)
              tile_size = 64;
          }

          // Select best kernel
          const char *kernel = cat_select_kernel(session->cached_tuner, M, N, K,
                                                 threads, tile_size);
          if (kernel) {
            node.selected_kernel = kernel;
          }
        }
      }
    }
  }

  return AGEE_OK;
}

/* ========================================================================== */
/* NUMA Node Assignment                                                       */
/* ========================================================================== */

int AssignNUMANodes(SessionImpl *session, GraphPlanImpl *plan) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  if (session->max_numa_nodes <= 1) {
    return AGEE_OK; // Single NUMA node or no NUMA support
  }

  // Strategy: Assign memory-intensive operations to different NUMA nodes
  // to maximize memory bandwidth utilization

  // Sort nodes by memory bandwidth requirements
  std::vector<ExecutionNode *> sorted_nodes;
  for (auto &node : plan->execution_nodes) {
    sorted_nodes.push_back(&node);
  }

  std::sort(sorted_nodes.begin(), sorted_nodes.end(),
            [](const ExecutionNode *a, const ExecutionNode *b) {
              return a->estimated_memory_bytes > b->estimated_memory_bytes;
            });

  // Round-robin assignment to NUMA nodes
  int current_numa_node = 0;
  for (auto *node : sorted_nodes) {
    node->numa_node = current_numa_node;
    current_numa_node = (current_numa_node + 1) % session->max_numa_nodes;

    // Assign tensors to the same NUMA node
    for (uint64_t tensor_id : node->input_tensors) {
      auto it = plan->tensors.find(tensor_id);
      if (it != plan->tensors.end()) {
        it->second.numa_node = node->numa_node;
      }
    }

    for (uint64_t tensor_id : node->output_tensors) {
      auto it = plan->tensors.find(tensor_id);
      if (it != plan->tensors.end()) {
        it->second.numa_node = node->numa_node;
      }
    }
  }

  if (session->config.verbose) {
    printf("[AGEE] NUMA assignment: %d nodes, %zu operations\n",
           session->max_numa_nodes, plan->execution_nodes.size());
  }

  return AGEE_OK;
}

/* ========================================================================== */
/* Thread Affinity Assignment                                                 */
/* ========================================================================== */

int AssignThreadAffinity(SessionImpl *session, GraphPlanImpl *plan) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  size_t num_threads = session->config.num_threads > 0
                           ? session->config.num_threads
                           : session->cpu_info.cores;

  // Simple strategy: Distribute operations evenly across threads
  size_t thread_idx = 0;
  for (auto &node : plan->execution_nodes) {
    node.thread_affinity = thread_idx;
    thread_idx = (thread_idx + 1) % num_threads;
  }

  // For parallel operations (GEMM), use all threads
  for (auto &node : plan->execution_nodes) {
    if (IsGEMMOperation(node.op_type)) {
      node.thread_affinity = 0; // Will use all threads
    }
  }

  return AGEE_OK;
}

} // namespace agee_internal

using namespace agee_internal;

/* ========================================================================== */
/* Public API - System Information                                            */
/* ========================================================================== */

extern "C" {

int agee_get_system_info(char **out_info) {
  if (!out_info) {
    return AGEE_ERR_INVALID_ARG;
  }

  if (!g_agee_state.initialized) {
    return AGEE_ERR_NOT_INITIALIZED;
  }

  char buffer[8192];
  int offset = 0;

  offset += snprintf(buffer + offset, sizeof(buffer) - offset,
                     "Adaptive Graph Execution Engine System Information:\n\n");

  // CPU Information
  offset +=
      snprintf(buffer + offset, sizeof(buffer) - offset,
               "CPU:\n"
               "  Cores: %d (logical: %d)\n"
               "  NUMA nodes: %d\n"
               "  AVX: %s\n"
               "  AVX2: %s\n"
               "  AVX-512: %s\n"
               "  AMX: %s\n\n",
               g_agee_state.cpu_info.cores, g_agee_state.cpu_info.logical_cores,
               g_agee_state.cpu_info.numa_nodes,
               g_agee_state.cpu_info.avx ? "Yes" : "No",
               g_agee_state.cpu_info.avx2 ? "Yes" : "No",
               g_agee_state.cpu_info.avx512 ? "Yes" : "No",
               g_agee_state.cpu_info.amx ? "Yes" : "No");

  // Cache Information
  offset += snprintf(buffer + offset, sizeof(buffer) - offset,
                     "Cache:\n"
                     "  L1d: %d KB\n"
                     "  L1i: %d KB\n"
                     "  L2: %d KB\n"
                     "  L3: %d KB\n\n",
                     g_agee_state.cpu_info.l1d_kb, g_agee_state.cpu_info.l1i_kb,
                     g_agee_state.cpu_info.l2_kb, g_agee_state.cpu_info.l3_kb);

  // Configuration
  offset += snprintf(
      buffer + offset, sizeof(buffer) - offset,
      "Configuration:\n"
      "  Fusion enabled: %s\n"
      "  NUMA optimization: %s\n"
      "  Prefetch enabled: %s\n"
      "  Adaptive tuning: %s\n"
      "  Memory pooling: %s\n"
      "  Huge pages: %s\n\n",
      g_agee_state.global_config.enable_fusion ? "Yes" : "No",
      g_agee_state.global_config.enable_numa_optimization ? "Yes" : "No",
      g_agee_state.global_config.enable_prefetch ? "Yes" : "No",
      g_agee_state.global_config.enable_adaptive_tuning ? "Yes" : "No",
      g_agee_state.global_config.enable_memory_pooling ? "Yes" : "No",
      g_agee_state.global_config.use_hugepages ? "Yes" : "No");

  // Component Status
  offset += snprintf(
      buffer + offset, sizeof(buffer) - offset,
      "Components:\n"
      "  Kernel Fusion Engine: %s\n"
      "  Operator Graph Runtime: %s\n"
      "  Vector Math Engine: %s\n"
      "  Profiler: %s\n"
      "  Kernel Dispatch: %s\n\n",
      g_agee_state.kfe_initialized ? "Initialized" : "Not initialized",
      g_agee_state.og_initialized ? "Initialized" : "Not initialized",
      g_agee_state.vmath_initialized ? "Initialized" : "Not initialized",
      g_agee_state.profiler_initialized ? "Initialized" : "Not initialized",
      g_agee_state.dispatch_initialized ? "Initialized" : "Not initialized");

  // Statistics
  offset += snprintf(buffer + offset, sizeof(buffer) - offset,
                     "Statistics:\n"
                     "  Active sessions: %zu\n"
                     "  Total graphs executed: %zu\n"
                     "  Total fusions performed: %zu\n",
                     g_agee_state.sessions.size(),
                     g_agee_state.total_graphs_executed.load(),
                     g_agee_state.total_fusions_performed.load());

  *out_info = strdup(buffer);
  return *out_info ? AGEE_OK : AGEE_ERR_NO_MEMORY;
}

int agee_get_session_stats(agee_session_t session,
                           agee_exec_stats_t *out_stats) {
  if (!session || !out_stats) {
    return AGEE_ERR_INVALID_ARG;
  }

  SessionImpl *impl = reinterpret_cast<SessionImpl *>(session);
  std::lock_guard<std::mutex> lock(impl->session_mutex);

  memcpy(out_stats, &impl->cumulative_stats, sizeof(agee_exec_stats_t));

  return AGEE_OK;
}

} // extern "C"