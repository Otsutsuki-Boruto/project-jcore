// advanced/src/agee_executor.cpp
#include "agee_internal.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>

#include "k_kernel_dispatch.h"
#include "vmath_engine.h"

namespace agee_internal {

/* ========================================================================== */
/* Execute Individual Node                                                    */
/* ========================================================================== */

int ExecuteNode(SessionImpl *session, GraphPlanImpl *plan,
                ExecutionNode *node) {
  if (!session || !plan || !node) {
    return AGEE_ERR_INVALID_ARG;
  }

  auto start = std::chrono::high_resolution_clock::now();

  int ret = og_execute_node(
    plan->original_graph,  // ← Changed from session->operator_graph
    node->og_node_id
);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  node->actual_time_ms = elapsed.count();
  node->is_executed = (ret == OG_OK);

  return (ret == OG_OK) ? AGEE_OK : AGEE_ERR_EXECUTION_FAILED;
}

/* ========================================================================== */
/* Execute Fusion Group                                                       */
/* ========================================================================== */

int ExecuteFusionGroup(SessionImpl *session, GraphPlanImpl *plan,
                         FusionGroup *group) {
  if (!session || !plan || !group) {
    fprintf(stderr, "[DEBUG] ExecuteFusionGroup: NULL parameter check failed\n");
    return AGEE_ERR_INVALID_ARG;
  }

  auto start = std::chrono::high_resolution_clock::now();

  og_graph_stats_t og_stats = {};
  int ret = og_execute_fusion_group(
      group->og_graph,
      &group->og_fusion_group,
      &og_stats);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  group->actual_time_ms = elapsed.count();
  group->is_executed = (ret == OG_OK);

  /* Mark all nodes in the fusion group as executed */
  for (uint64_t og_nid : group->node_ids) {
    auto node_it = std::find_if(
        plan->execution_nodes.begin(), plan->execution_nodes.end(),
        [og_nid](const ExecutionNode &n) { return n.og_node_id == og_nid; });

    if (node_it != plan->execution_nodes.end()) {
      ExecutionNode *node = const_cast<ExecutionNode *>(&(*node_it));
      node->is_executed = (ret == OG_OK);
    }
  }

  return (ret == OG_OK) ? AGEE_OK : AGEE_ERR_INTERNAL;
}


/* ========================================================================== */
/* Execute Entire Plan                                                        */
/* ========================================================================== */

int ExecutePlanInternal(SessionImpl *session, GraphPlanImpl *plan,
                        agee_exec_stats_t *stats) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Reset execution state
  for (auto &node : plan->execution_nodes) {
    node.is_executed = false;
    node.actual_time_ms = 0.0;
  }

  for (auto &group : plan->fusion_groups) {
    group.is_executed = false;
    group.actual_time_ms = 0.0;
  }

  // Build set of nodes covered by fusion groups
  std::unordered_set<uint64_t> fused_nodes;
  for (const auto &group : plan->fusion_groups) {
    for (uint64_t nid : group.node_ids) {
      fused_nodes.insert(nid);
    }
  }

  // Execute fusion groups first (if any exist and have valid patterns)
  for (auto &group : plan->fusion_groups) {
    // Only execute if we have a recognized fusion pattern
    if (group.pattern != OG_PATTERN_NONE && group.node_ids.size() > 0) {

      int ret = ExecuteFusionGroup(session, plan, &group);

      if (ret != AGEE_OK && session->config.verbose) {
        // Don't fail - fall back to executing nodes individually
        fused_nodes.clear(); // Clear fused nodes so they execute individually
      }
    }
  }

  // Execute remaining non-fused nodes
  for (const uint64_t node_id : plan->execution_order) {
    // Skip if already executed as part of fusion
    if (fused_nodes.find(node_id) != fused_nodes.end()) {
      auto node_it = std::find_if(
          plan->execution_nodes.begin(), plan->execution_nodes.end(),
          [node_id](const ExecutionNode &n) { return n.node_id == node_id; });
      if (node_it != plan->execution_nodes.end() &&
          const_cast<ExecutionNode *>(&(*node_it))->is_executed) {
        continue; // Already executed in fusion
      }
      // If not executed, fall through to execute individually
    }

    auto node_it = std::find_if(
        plan->execution_nodes.begin(), plan->execution_nodes.end(),
        [node_id](const ExecutionNode &n) { return n.node_id == node_id; });

    if (node_it == plan->execution_nodes.end()) {
      continue;
    }

    ExecutionNode *node = const_cast<ExecutionNode *>(&(*node_it));

    // Skip if already executed
    if (node->is_executed) {
      continue;
    }

    // Check dependencies
    bool deps_satisfied = true;
    for (uint64_t dep_id : node->dependencies) {
      auto dep_it = std::find_if(
          plan->execution_nodes.begin(), plan->execution_nodes.end(),
          [dep_id](const ExecutionNode &n) { return n.node_id == dep_id; });

      if (dep_it != plan->execution_nodes.end() && !dep_it->is_executed) {
        deps_satisfied = false;
        break;
      }
    }

    if (!deps_satisfied) {
      if (session->config.verbose) {
        fprintf(stderr, "[AGEE] Warning: Node %lu deps not satisfied\n",
                node_id);
      }
      continue;
    }

    int ret = ExecuteNode(session, plan, node);
    if (ret != AGEE_OK && session->config.verbose) {
      fprintf(stderr, "[AGEE] Warning: Node %lu execution failed\n", node_id);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> total_elapsed =
      end_time - start_time;

  // Collect statistics
  if (stats) {
    memset(stats, 0, sizeof(agee_exec_stats_t));

    stats->total_execution_time_ms = total_elapsed.count();
    stats->total_operations = plan->execution_nodes.size();
    stats->fused_operations = plan->fused_operations;
    stats->memory_allocated_bytes = plan->peak_memory_bytes;

    // Sum up actual execution times
    for (const auto &node : plan->execution_nodes) {
      if (node.is_executed) {
        stats->fusion_time_ms += node.actual_time_ms;
      }
    }

    // Compute GFLOPS - avoid double counting fused nodes
    double total_flops = 0.0;

    // First, add FLOPs from executed fusion groups
    for (const auto &group : plan->fusion_groups) {
      if (group.is_executed) {
        total_flops += group.estimated_flops;
      }
    }

    // Build set of fused node IDs to avoid double counting
    std::unordered_set<uint64_t> fused_node_ids;
    for (const auto &group : plan->fusion_groups) {
      if (group.is_executed) {
        fused_node_ids.insert(group.node_ids.begin(), group.node_ids.end());
      }
    }

    // Then add FLOPs from non-fused nodes only
    for (const auto &node : plan->execution_nodes) {
      if (node.is_executed && fused_node_ids.find(node.node_id) == fused_node_ids.end()) {
        total_flops += node.estimated_flops;
      }
    }

    if (stats->total_execution_time_ms > 0.0) {
      stats->achieved_gflops =
          ComputeGFLOPS(total_flops, stats->total_execution_time_ms);
    }

    stats->memory_saved_bytes = 0;
    for (const auto &group : plan->fusion_groups) {
      if (group.is_executed) {
        stats->memory_saved_bytes += group.estimated_memory_saved;
      }
    }

    if (plan->estimated_total_time_ms > 0.0) {
      stats->fusion_speedup =
          plan->estimated_total_time_ms / stats->total_execution_time_ms;
    }
  }

  // Update session statistics
  session->total_executions.fetch_add(1);
  g_agee_state.total_graphs_executed.fetch_add(1);

  size_t executed_fusions = 0;
  for (const auto &group : plan->fusion_groups) {
    if (group.is_executed) {
      executed_fusions++;
    }
  }
  g_agee_state.total_fusions_performed.fetch_add(executed_fusions);

  return AGEE_OK;
}

} // namespace agee_internal

using namespace agee_internal;

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

extern "C" {

int agee_execute_plan(agee_session_t session, agee_graph_plan_t plan,
                      agee_exec_stats_t *stats) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  if (!g_agee_state.initialized) {
    return AGEE_ERR_NOT_INITIALIZED;
  }

  SessionImpl *sess_impl = reinterpret_cast<SessionImpl *>(session);
  GraphPlanImpl *plan_impl = reinterpret_cast<GraphPlanImpl *>(plan);

  return ExecutePlanInternal(sess_impl, plan_impl, stats);
}

int agee_execute_plan_with_tensors(agee_session_t session,
                                   agee_graph_plan_t plan, void **input_tensors,
                                   size_t num_inputs, void **output_tensors,
                                   size_t num_outputs,
                                   agee_exec_stats_t *stats) {
  if (!session || !plan || !input_tensors || !output_tensors) {
    return AGEE_ERR_INVALID_ARG;
  }

  SessionImpl *sess_impl = reinterpret_cast<SessionImpl *>(session);
  GraphPlanImpl *plan_impl = reinterpret_cast<GraphPlanImpl *>(plan);

  // Assign input tensors
  size_t input_idx = 0;
  for (auto &pair : plan_impl->tensors) {
    if (!pair.second.is_constant && input_idx < num_inputs) {
      pair.second.data = input_tensors[input_idx++];
    }
  }

  // Execute plan
  int ret = ExecutePlanInternal(sess_impl, plan_impl, stats);

  // Copy output tensors
  size_t output_idx = 0;
  for (const auto &pair : plan_impl->tensors) {
    if (output_idx >= num_outputs)
      break;

    // Check if this is an output tensor (not consumed by other nodes)
    bool is_output = true;
    for (const auto &node : plan_impl->execution_nodes) {
      if (std::find(node.input_tensors.begin(), node.input_tensors.end(),
                    pair.first) != node.input_tensors.end()) {
        is_output = false;
        break;
      }
    }

    if (is_output && pair.second.data) {
      memcpy(output_tensors[output_idx++], pair.second.data,
             pair.second.size_bytes);
    }
  }

  return ret;
}

int agee_execute_batch(agee_session_t session, agee_graph_plan_t *plans,
                       size_t num_plans, agee_exec_stats_t *stats_array) {
  if (!session || !plans || num_plans == 0) {
    return AGEE_ERR_INVALID_ARG;
  }

  SessionImpl *sess_impl = reinterpret_cast<SessionImpl *>(session);

  // Execute plans in parallel if thread scheduler is available
  if (sess_impl->thread_scheduler) {
    bool success =
        sess_impl->thread_scheduler->ParallelFor(num_plans, [&](size_t i) {
          GraphPlanImpl *plan = reinterpret_cast<GraphPlanImpl *>(plans[i]);
          agee_exec_stats_t *stats = stats_array ? &stats_array[i] : nullptr;
          ExecutePlanInternal(sess_impl, plan, stats);
        });

    return success ? AGEE_OK : AGEE_ERR_INTERNAL;
  } else {
    // Sequential fallback
    for (size_t i = 0; i < num_plans; ++i) {
      GraphPlanImpl *plan = reinterpret_cast<GraphPlanImpl *>(plans[i]);
      agee_exec_stats_t *stats = stats_array ? &stats_array[i] : nullptr;
      int ret = ExecutePlanInternal(sess_impl, plan, stats);
      if (ret != AGEE_OK) {
        return ret;
      }
    }
  }

  return AGEE_OK;
}

} // extern "C"