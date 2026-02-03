// advanced/src/agee_graph_analysis.cpp
#include "agee_internal.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <queue>

namespace agee_internal {

/* ========================================================================== */
/* Graph Analysis - Convert OG graph to execution plan                       */
/* ========================================================================== */

int AnalyzeGraph(SessionImpl *session, og_graph_t graph, GraphPlanImpl *plan) {
  if (!session || !graph || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  plan->original_graph = graph;

  // Get graph statistics
  og_graph_stats_t og_stats = {};
  if (og_get_graph_stats(graph, &og_stats) != OG_OK) {
    fprintf(stderr, "[AGEE] Failed to get graph statistics\n");
    return AGEE_ERR_GRAPH_INVALID;
  }

  plan->total_operations = og_stats.total_nodes;

  if (session->config.verbose) {
    printf("[AGEE] Analyzing graph: %zu nodes, %zu tensors\n",
           og_stats.total_nodes, og_stats.total_tensors);
  }

  // Get execution order
  uint64_t *exec_order = new (std::nothrow) uint64_t[og_stats.total_nodes];
  if (!exec_order) {
    return AGEE_ERR_NO_MEMORY;
  }

  size_t num_nodes = 0;
  int ret = og_get_execution_order(graph, exec_order, og_stats.total_nodes,
                                   &num_nodes);
  if (ret != OG_OK) {
    delete[] exec_order;
    fprintf(stderr, "[AGEE] Failed to get execution order\n");
    return AGEE_ERR_GRAPH_INVALID;
  }

  // Convert nodes to execution nodes
  plan->execution_nodes.reserve(num_nodes);

  for (size_t i = 0; i < num_nodes; ++i) {
    uint64_t node_id = exec_order[i];
    og_node_t og_node = {};

    if (og_get_node_info(graph, node_id, &og_node) != OG_OK) {
      delete[] exec_order;
      return AGEE_ERR_GRAPH_INVALID;
    }

    ExecutionNode exec_node;
    exec_node.node_id = i;        // AGEE's sequential ID
    exec_node.og_node_id = node_id;  // OG's actual node ID
    exec_node.op_type = og_node.type;

    // Copy input tensors
    for (size_t j = 0; j < og_node.num_inputs; ++j) {
      exec_node.input_tensors.push_back(og_node.input_ids[j]);
    }

    // Copy output tensors
    for (size_t j = 0; j < og_node.num_outputs; ++j) {
      exec_node.output_tensors.push_back(og_node.output_ids[j]);
    }

    // Estimate FLOPs for GEMM operations
    if (IsGEMMOperation(og_node.type) && og_node.num_inputs >= 2) {
      og_tensor_t t1 = {}, t2 = {};
      if (og_get_tensor_info(graph, og_node.input_ids[0], &t1) == OG_OK &&
          og_get_tensor_info(graph, og_node.input_ids[1], &t2) == OG_OK) {

        if (t1.ndim >= 2 && t2.ndim >= 2) {
          size_t M = t1.shape[0];
          size_t K = t1.shape[1];
          size_t N = t2.shape[1];
          exec_node.estimated_flops = 2 * M * N * K;
          exec_node.estimated_memory_bytes =
              (M * K + K * N + M * N) * sizeof(float);
        }
      }
    }

    // Store tensor descriptors
    for (size_t j = 0; j < og_node.num_inputs; ++j) {
      og_tensor_t tensor = {};
      if (og_get_tensor_info(graph, og_node.input_ids[j], &tensor) == OG_OK) {
        if (plan->tensors.find(tensor.id) == plan->tensors.end()) {
          TensorDescriptor desc;
          desc.tensor_id = tensor.id;
          desc.ndim = tensor.ndim;
          desc.size_bytes = tensor.size_bytes;
          memcpy(desc.shape, tensor.shape, sizeof(desc.shape));
          memcpy(desc.strides, tensor.strides, sizeof(desc.strides));
          desc.is_constant = tensor.is_constant;
          plan->tensors[tensor.id] = desc;
        }
      }
    }

    // At end of loop in AnalyzeGraph:
    plan->execution_nodes.push_back(exec_node);
    plan->execution_order.push_back(exec_node.og_node_id);  // ← Change from node_id to og_node_id
  }

  delete[] exec_order;

  if (session->config.verbose) {
    printf("[AGEE] Created %zu execution nodes\n",
           plan->execution_nodes.size());
  }

  return AGEE_OK;
}

/* ========================================================================== */
/* Fusion Opportunity Detection                                               */
/* ========================================================================== */

int DetectFusionOpportunities(GraphPlanImpl *plan) {
  if (!plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  // Skip if no nodes to fuse
  if (plan->execution_nodes.empty()) {
    return AGEE_OK;
  }

  // Detect fusion patterns using Operator Graph Runtime
  og_fusion_group_t *og_groups =
      new (std::nothrow) og_fusion_group_t[plan->execution_nodes.size()];
  if (!og_groups) {
    return AGEE_ERR_NO_MEMORY;
  }

  size_t num_groups = 0;
  int ret =
      og_detect_fusion_patterns(plan->original_graph, og_groups,
                                plan->execution_nodes.size(), &num_groups);

  // If fusion detection fails, just continue without fusion (non-fatal)
  if (ret != OG_OK) {
    delete[] og_groups;
    if (g_agee_state.global_config.verbose) {
      fprintf(stderr, "[AGEE] Warning: Fusion detection returned error %d\n",
              ret);
    }
    return AGEE_OK; // Non-fatal, continue without fusion
  }

  // Convert OG fusion groups to internal representation
  plan->fusion_groups.reserve(num_groups);

  for (size_t i = 0; i < num_groups; ++i) {
    FusionGroup group;
    group.group_id = og_groups[i].group_id;
    group.pattern = og_groups[i].pattern;
    group.estimated_memory_saved = og_groups[i].estimated_memory_saved_bytes;
    group.estimated_speedup = og_groups[i].estimated_speedup;

    // ===== FIX: Store OG references for execution =====
    group.og_graph = plan->original_graph;
    group.og_fusion_group = og_groups[i];
    // ===== END FIX =====

    // Copy node IDs
    for (size_t j = 0; j < og_groups[i].num_nodes; ++j) {
      group.node_ids.push_back(og_groups[i].node_ids[j]);
    }

    // ===== FIX: Aggregate FLOPs from all nodes in fusion group =====
    for (uint64_t og_nid : group.node_ids) {
      auto it = std::find_if(
          plan->execution_nodes.begin(), plan->execution_nodes.end(),
          [og_nid](const ExecutionNode &n) { return n.og_node_id == og_nid; });

      if (it != plan->execution_nodes.end()) {
        group.estimated_flops += it->estimated_flops;
      }
    }

    // Calculate memory savings if OG didn't provide it
    if (group.estimated_memory_saved == 0 && group.node_ids.size() > 1) {
      for (size_t j = 0; j < group.node_ids.size() - 1; ++j) {
        uint64_t og_nid = group.node_ids[j];
        auto node_it = std::find_if(
            plan->execution_nodes.begin(), plan->execution_nodes.end(),
            [og_nid](const ExecutionNode &n) { return n.og_node_id == og_nid; });

        if (node_it != plan->execution_nodes.end()) {
          // Check output tensors
          for (uint64_t out_tensor_id : node_it->output_tensors) {
            bool is_intermediate = true;

            // Check if used outside fusion group
            for (const auto &other_node : plan->execution_nodes) {
              if (std::find(group.node_ids.begin(), group.node_ids.end(),
                           other_node.node_id) != group.node_ids.end()) {
                continue;
              }

              if (std::find(other_node.input_tensors.begin(),
                           other_node.input_tensors.end(),
                           out_tensor_id) != other_node.input_tensors.end()) {
                is_intermediate = false;
                break;
              }
            }

            if (is_intermediate) {
              auto tensor_it = plan->tensors.find(out_tensor_id);
              if (tensor_it != plan->tensors.end()) {
                group.estimated_memory_saved += tensor_it->second.size_bytes;
              }
            }
          }
        }
      }
    }
    // ===== END FIX =====

    // Detect activation type if applicable
    if (og_groups[i].pattern == OG_PATTERN_GEMM_BIAS_ACTIVATION ||
        og_groups[i].pattern == OG_PATTERN_GEMM_RESIDUAL_ACTIVATION) {

      // Find activation node
      for (uint64_t og_nid : group.node_ids) {
        auto it = std::find_if(
            plan->execution_nodes.begin(), plan->execution_nodes.end(),
            [og_nid](const ExecutionNode &n) { return n.og_node_id == og_nid; });

        if (it != plan->execution_nodes.end() &&
            IsActivationOperation(it->op_type)) {
          group.activation = ConvertActivationType(it->op_type);
          break;
        }
      }
    }

    plan->fusion_groups.push_back(group);
    plan->fused_operations += og_groups[i].num_nodes;
  }

  delete[] og_groups;

  if (g_agee_state.global_config.verbose) {
    printf("[AGEE] Detected %zu fusion groups, %zu ops fused\n",
           plan->fusion_groups.size(), plan->fused_operations);
  }

  return AGEE_OK;
}

/* ========================================================================== */
/* Build Execution Order with Dependencies                                    */
/* ========================================================================== */

int BuildExecutionOrder(GraphPlanImpl *plan) {
  if (!plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  // Build dependency graph
  std::unordered_map<uint64_t, std::vector<uint64_t>> dependencies;
  std::unordered_map<uint64_t, int> in_degree;

  // Initialize
  for (const auto &node : plan->execution_nodes) {
    in_degree[node.node_id] = 0;
    dependencies[node.node_id] = {};
  }

  // Build edges based on tensor producer-consumer relationships
  std::unordered_map<uint64_t, uint64_t>
      tensor_producer; // tensor_id -> node_id

  for (auto &node : plan->execution_nodes) {
    // Register this node as producer of its output tensors
    for (uint64_t tensor_id : node.output_tensors) {
      tensor_producer[tensor_id] = node.node_id;
    }
  }

  for (auto &node : plan->execution_nodes) {
    // For each input tensor, find its producer
    for (uint64_t tensor_id : node.input_tensors) {
      auto it = tensor_producer.find(tensor_id);
      if (it != tensor_producer.end() && it->second != node.node_id) {
        uint64_t producer_id = it->second;
        dependencies[node.node_id].push_back(producer_id);
        in_degree[node.node_id]++;
      }
    }

    // Store dependencies in node
    auto &exec_node = const_cast<ExecutionNode &>(node);
    exec_node.dependencies = dependencies[node.node_id];
  }

  // Topological sort using Kahn's algorithm
  std::queue<uint64_t> ready_queue;
  std::vector<uint64_t> sorted_order;

  // Find all nodes with no dependencies
  for (const auto &pair : in_degree) {
    if (pair.second == 0) {
      ready_queue.push(pair.first);
    }
  }

  while (!ready_queue.empty()) {
    uint64_t node_id = ready_queue.front();
    ready_queue.pop();
    sorted_order.push_back(node_id);

    // Find all nodes that depend on this node
    for (auto &node : plan->execution_nodes) {
      auto it = std::find(node.dependencies.begin(), node.dependencies.end(),
                          node_id);
      if (it != node.dependencies.end()) {
        in_degree[node.node_id]--;
        if (in_degree[node.node_id] == 0) {
          ready_queue.push(node.node_id);
        }
      }
    }
  }

  // Check for cycles
  if (sorted_order.size() != plan->execution_nodes.size()) {
    fprintf(stderr, "[AGEE] Cycle detected in graph\n");
    return AGEE_ERR_GRAPH_INVALID;
  }

  plan->execution_order = sorted_order;

  return AGEE_OK;
}

} // namespace agee_internal

using namespace agee_internal;

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

extern "C" {

int agee_create_plan_from_graph(agee_session_t session, void *graph,
                                agee_graph_plan_t *out_plan) {
  if (!session || !graph || !out_plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  if (!g_agee_state.initialized) {
    return AGEE_ERR_NOT_INITIALIZED;
  }

  SessionImpl *sess_impl = reinterpret_cast<SessionImpl *>(session);
  og_graph_t og_graph = reinterpret_cast<og_graph_t>(graph);

  // Allocate plan
  GraphPlanImpl *plan = new (std::nothrow) GraphPlanImpl();
  if (!plan) {
    return AGEE_ERR_NO_MEMORY;
  }

  plan->owning_session = sess_impl;

  // Analyze graph
  int ret = AnalyzeGraph(sess_impl, og_graph, plan);
  if (ret != AGEE_OK) {
    delete plan;
    return ret;
  }

  // Detect fusion opportunities
  if (sess_impl->config.enable_fusion) {
    ret = DetectFusionOpportunities(plan);
    if (ret != AGEE_OK && sess_impl->config.verbose) {
      fprintf(stderr, "[AGEE] Warning: Fusion detection failed\n");
    }
  }

  // Build execution order with dependencies
  ret = BuildExecutionOrder(plan);
  if (ret != AGEE_OK) {
    delete plan;
    return ret;
  }

  // Predict memory usage
  ret = PredictMemoryUsage(plan);
  if (ret != AGEE_OK && sess_impl->config.verbose) {
    fprintf(stderr, "[AGEE] Warning: Memory prediction failed\n");
  }

  // Schedule operations
  ret = ScheduleOperations(sess_impl, plan);
  if (ret != AGEE_OK && sess_impl->config.verbose) {
    fprintf(stderr, "[AGEE] Warning: Scheduling failed\n");
  }

  // Register plan with session
  {
    std::lock_guard<std::mutex> lock(sess_impl->session_mutex);
    sess_impl->active_plans.push_back(plan);
  }

  plan->is_optimized = true;
  *out_plan = reinterpret_cast<agee_graph_plan_t>(plan);

  if (sess_impl->config.verbose) {
    printf("[AGEE] Plan created: %zu ops, %zu fused, peak mem: %.2f MB\n",
           plan->total_operations, plan->fused_operations,
           plan->peak_memory_bytes / (1024.0 * 1024.0));
  }

  return AGEE_OK;
}

int agee_optimize_plan(agee_session_t session, agee_graph_plan_t plan) {
  if (!session || !plan) {
    return AGEE_ERR_INVALID_ARG;
  }

  SessionImpl *sess_impl = reinterpret_cast<SessionImpl *>(session);
  GraphPlanImpl *plan_impl = reinterpret_cast<GraphPlanImpl *>(plan);

  std::lock_guard<std::mutex> lock(plan_impl->plan_mutex);

  // Re-run fusion detection
  int ret = DetectFusionOpportunities(plan_impl);
  if (ret != AGEE_OK) {
    return ret;
  }

  // Re-schedule
  ret = ScheduleOperations(sess_impl, plan_impl);
  if (ret != AGEE_OK) {
    return ret;
  }

  plan_impl->is_optimized = true;

  return AGEE_OK;
}

  void agee_destroy_plan(agee_graph_plan_t plan) {
  if (!plan) {
    return;
  }

  GraphPlanImpl *impl = reinterpret_cast<GraphPlanImpl *>(plan);

  // CRITICAL: Find and unregister from the owning session
  SessionImpl *session = impl->owning_session;

  if (session) {
    std::lock_guard<std::mutex> lock(session->session_mutex);

    // Remove from active_plans list
    auto it = std::find(session->active_plans.begin(),
                       session->active_plans.end(),
                       impl);
    if (it != session->active_plans.end()) {
      session->active_plans.erase(it);

      if (session->config.verbose) {
        printf("[AGEE] Unregistered plan from session (remaining: %zu)\n",
               session->active_plans.size());
      }
    }
  }

  // Free any allocated kernel parameters
  for (auto &node : impl->execution_nodes) {
    if (node.kernel_params) {
      free(node.kernel_params);
      node.kernel_params = nullptr;
    }
  }

  // Note: Do not destroy original_graph as it's owned by the caller
  delete impl;
}

int agee_get_plan_info(agee_graph_plan_t plan, char **out_info) {
  if (!plan || !out_info) {
    return AGEE_ERR_INVALID_ARG;
  }

  GraphPlanImpl *impl = reinterpret_cast<GraphPlanImpl *>(plan);

  char buffer[4096];
  snprintf(buffer, sizeof(buffer),
           "Graph Execution Plan:\n"
           "  Total operations: %zu\n"
           "  Fused operations: %zu\n"
           "  Fusion groups: %zu\n"
           "  Peak memory: %.2f MB\n"
           "  Estimated time: %.2f ms\n"
           "  Optimized: %s\n",
           impl->total_operations, impl->fused_operations,
           impl->fusion_groups.size(),
           impl->peak_memory_bytes / (1024.0 * 1024.0),
           impl->estimated_total_time_ms, impl->is_optimized ? "Yes" : "No");

  *out_info = strdup(buffer);
  return *out_info ? AGEE_OK : AGEE_ERR_NO_MEMORY;
}

} // extern "C"