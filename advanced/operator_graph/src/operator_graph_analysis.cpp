// advanced/operator_graph/src/operator_graph_analysis.cpp
/**
 * @file operator_graph_analysis.cpp
 * @brief Graph analysis - topological sort, cycle detection, pattern matching
 */

#include "operator_graph_internal.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <queue>

using namespace og_internal;

/* ========================================================================== */
/* Topological Sort                                                           */
/* ========================================================================== */

namespace og_internal
{

  int TopologicalSort(GraphImpl *graph)
  {
    graph->topological_order.clear();

    // Kahn's algorithm using in-degrees
    std::queue<uint64_t> zero_indegree_queue;
    std::unordered_map<uint64_t, int> temp_indegree;

    // Initialize in-degrees
    for (auto &node_pair : graph->nodes)
    {
      NodeImpl *node = node_pair.second.get();
      temp_indegree[node->descriptor.id] = node->in_degree;

      if (node->in_degree == 0)
      {
        zero_indegree_queue.push(node->descriptor.id);
      }
    }

    // Process nodes with zero in-degree
    while (!zero_indegree_queue.empty())
    {
      uint64_t node_id = zero_indegree_queue.front();
      zero_indegree_queue.pop();

      graph->topological_order.push_back(node_id);

      NodeImpl *node = GetNode(graph, node_id);
      if (!node)
        continue;

      // Reduce in-degree of successors
      for (uint64_t succ_id : node->successor_nodes)
      {
        temp_indegree[succ_id]--;
        if (temp_indegree[succ_id] == 0)
        {
          zero_indegree_queue.push(succ_id);
        }
      }
    }

    // If not all nodes processed, there's a cycle
    if (graph->topological_order.size() != graph->nodes.size())
    {
      return OG_ERR_CYCLE_DETECTED;
    }

    return OG_OK;
  }

  /* ========================================================================== */
  /* Cycle Detection                                                            */
  /* ========================================================================== */

  bool HasCycle(GraphImpl *graph)
  {
    std::unordered_set<uint64_t> visited;
    std::unordered_set<uint64_t> rec_stack;

    std::function<bool(uint64_t)> dfs = [&](uint64_t node_id) -> bool
    {
      visited.insert(node_id);
      rec_stack.insert(node_id);

      NodeImpl *node = GetNode(graph, node_id);

      if (!node) {
        return false;
      }

      if (g_og_state.config.verbose)
      {
        std::cout << "[OG] DFS visiting Node" << node_id
                  << " (successors: " << node->successor_nodes.size() << ")" << std::endl;
      }

      size_t num_successors = node->successor_nodes.size();

      for (uint64_t succ_id : node->successor_nodes)
      {
        if (rec_stack.find(succ_id) != rec_stack.end())
        {
          if (g_og_state.config.verbose)
          {
            std::cerr << "[OG] CYCLE DETECTED: Node" << node_id
                      << " -> Node" << succ_id << " (already in recursion stack)" << std::endl;
          }
          return true; // Cycle detected
        }

        if (visited.find(succ_id) == visited.end())
        {
          if (dfs(succ_id))
          {
            return true;
          }
        }
      }

      rec_stack.erase(node_id);
      return false;
    };

    // Safety check
    if (graph->nodes.empty()) {
      return false;
    }

    try {
      size_t node_count = 0;
      for (auto &node_pair : graph->nodes) {
        uint64_t node_id = node_pair.first;

        if (visited.find(node_id) == visited.end()) {
          if (dfs(node_id)) {
            return true;
          }
        }
        node_count++;
      }
    } catch (const std::exception &e) {
      return false;
    } catch (...) {
      return false;
    }
    if (g_og_state.config.verbose)
    {
      std::cout << "[OG] No cycles detected" << std::endl;
    }

    return false;
  }

  /* ========================================================================== */
  /* Pattern Detection - Main Orchestrator                                      */
  /* ========================================================================== */

  int AnalyzeAndDetectPatterns(GraphImpl *graph)
  {
    if (!g_og_state.config.enable_pattern_matching)
    {
      return OG_OK;
    }

    graph->fusion_groups.clear();
    size_t patterns_found = 0;

    // Traverse nodes in topological order
    for (uint64_t node_id : graph->topological_order)
    {
      NodeImpl *node = GetNode(graph, node_id);
      if (!node)
        continue;

      og_fusion_group_t group = {};

      // Try to detect various patterns starting from this node
      bool pattern_found = false;

      // Pattern 1: GEMM + Bias + Activation
      if (node->descriptor.type == OG_OP_GEMM)
      {
        if (DetectGemmBiasActivationPattern(graph, node_id, &group))
        {
          pattern_found = true;
        }
        else if (DetectGemmBiasPattern(graph, node_id, &group))
        {
          pattern_found = true;
        }
        else if (DetectGemmResidualPattern(graph, node_id, &group))
        {
          pattern_found = true;
        }
      }

      if (pattern_found)
      {
        group.group_id = graph->next_fusion_group_id.fetch_add(1, std::memory_order_relaxed);
        auto group_impl = std::make_unique<FusionGroupImpl>(group);
        graph->fusion_groups[group.group_id] = std::move(group_impl);
        patterns_found++;

        if (g_og_state.config.verbose)
        {
          std::cout << "[OG] Detected fusion pattern: " << og_pattern_name(group.pattern)
                    << " (group_id=" << group.group_id
                    << ", nodes=" << group.num_nodes << ")" << std::endl;
        }
      }
    }

    graph->stats.fusion_groups_detected = patterns_found;

    if (g_og_state.config.verbose)
    {
      std::cout << "[OG] Pattern detection complete: " << patterns_found << " fusion groups found" << std::endl;
    }

    return OG_OK;
  }

  /* ========================================================================== */
  /* Pattern Detection - GEMM + Bias                                            */
  /* ========================================================================== */

  bool DetectGemmBiasPattern(GraphImpl *graph, uint64_t node_id, og_fusion_group_t *out_group)
  {
    NodeImpl *gemm_node = GetNode(graph, node_id);
    if (!gemm_node || gemm_node->descriptor.type != OG_OP_GEMM)
    {
      return false;
    }

    // Check if GEMM has exactly one successor
    if (gemm_node->successor_nodes.size() != 1)
    {
      return false;
    }

    uint64_t succ_id = gemm_node->successor_nodes[0];
    NodeImpl *bias_node = GetNode(graph, succ_id);

    if (!bias_node || bias_node->descriptor.type != OG_OP_BIAS_ADD)
    {
      return false;
    }

    // Found pattern: GEMM -> Bias
    out_group->pattern = OG_PATTERN_GEMM_BIAS;
    out_group->node_ids[0] = node_id;
    out_group->node_ids[1] = succ_id;
    out_group->num_nodes = 2;

    // Estimate memory savings (one intermediate buffer eliminated)
    if (!gemm_node->output_tensor_ids.empty())
    {
      TensorImpl *output = GetTensor(graph, gemm_node->output_tensor_ids[0]);
      if (output)
      {
        out_group->estimated_memory_saved_bytes = output->descriptor.size_bytes;
      }
    }

    out_group->estimated_speedup = 1.2; // 20% speedup estimate

    return true;
  }

  /* ========================================================================== */
  /* Pattern Detection - GEMM + Bias + Activation                               */
  /* ========================================================================== */

  bool DetectGemmBiasActivationPattern(GraphImpl *graph, uint64_t node_id,
                                       og_fusion_group_t *out_group)
  {
    NodeImpl *gemm_node = GetNode(graph, node_id);
    if (!gemm_node || gemm_node->descriptor.type != OG_OP_GEMM)
    {
      return false;
    }

    // Check if GEMM has exactly one successor
    if (gemm_node->successor_nodes.size() != 1)
    {
      return false;
    }

    uint64_t bias_id = gemm_node->successor_nodes[0];
    NodeImpl *bias_node = GetNode(graph, bias_id);

    if (!bias_node || bias_node->descriptor.type != OG_OP_BIAS_ADD)
    {
      return false;
    }

    // Check if Bias has exactly one successor
    if (bias_node->successor_nodes.size() != 1)
    {
      return false;
    }

    uint64_t act_id = bias_node->successor_nodes[0];
    NodeImpl *act_node = GetNode(graph, act_id);

    if (!act_node || !IsActivationOp(act_node->descriptor.type))
    {
      return false;
    }

    // Found pattern: GEMM -> Bias -> Activation
    out_group->pattern = OG_PATTERN_GEMM_BIAS_ACTIVATION;
    out_group->node_ids[0] = node_id;
    out_group->node_ids[1] = bias_id;
    out_group->node_ids[2] = act_id;
    out_group->num_nodes = 3;

    // Estimate memory savings (two intermediate buffers eliminated)
    size_t total_saved = 0;
    if (!gemm_node->output_tensor_ids.empty())
    {
      TensorImpl *output = GetTensor(graph, gemm_node->output_tensor_ids[0]);
      if (output)
      {
        total_saved += output->descriptor.size_bytes;
      }
    }
    if (!bias_node->output_tensor_ids.empty())
    {
      TensorImpl *output = GetTensor(graph, bias_node->output_tensor_ids[0]);
      if (output)
      {
        total_saved += output->descriptor.size_bytes;
      }
    }

    out_group->estimated_memory_saved_bytes = total_saved;
    out_group->estimated_speedup = 1.5; // 50% speedup estimate

    return true;
  }

  /* ========================================================================== */
  /* Pattern Detection - GEMM + Residual + Activation                           */
  /* ========================================================================== */

  bool DetectGemmResidualPattern(GraphImpl *graph, uint64_t node_id,
                                 og_fusion_group_t *out_group)
  {
    NodeImpl *gemm_node = GetNode(graph, node_id);
    if (!gemm_node || gemm_node->descriptor.type != OG_OP_GEMM)
    {
      return false;
    }

    // Look for pattern: GEMM -> Bias -> ElementwiseAdd(residual) -> Activation
    if (gemm_node->successor_nodes.size() != 1)
    {
      return false;
    }

    uint64_t bias_id = gemm_node->successor_nodes[0];
    NodeImpl *bias_node = GetNode(graph, bias_id);

    if (!bias_node || bias_node->descriptor.type != OG_OP_BIAS_ADD)
    {
      return false;
    }

    if (bias_node->successor_nodes.size() != 1)
    {
      return false;
    }

    uint64_t add_id = bias_node->successor_nodes[0];
    NodeImpl *add_node = GetNode(graph, add_id);

    if (!add_node || add_node->descriptor.type != OG_OP_ELEMENTWISE_ADD)
    {
      return false;
    }

    if (add_node->successor_nodes.size() != 1)
    {
      return false;
    }

    uint64_t act_id = add_node->successor_nodes[0];
    NodeImpl *act_node = GetNode(graph, act_id);

    if (!act_node || !IsActivationOp(act_node->descriptor.type))
    {
      return false;
    }

    // Found pattern: GEMM -> Bias -> ElementwiseAdd -> Activation
    out_group->pattern = OG_PATTERN_GEMM_RESIDUAL_ACTIVATION;
    out_group->node_ids[0] = node_id;
    out_group->node_ids[1] = bias_id;
    out_group->node_ids[2] = add_id;
    out_group->node_ids[3] = act_id;
    out_group->num_nodes = 4;

    // Estimate memory savings (three intermediate buffers eliminated)
    size_t total_saved = 0;
    if (!gemm_node->output_tensor_ids.empty())
    {
      TensorImpl *output = GetTensor(graph, gemm_node->output_tensor_ids[0]);
      if (output)
        total_saved += output->descriptor.size_bytes;
    }
    if (!bias_node->output_tensor_ids.empty())
    {
      TensorImpl *output = GetTensor(graph, bias_node->output_tensor_ids[0]);
      if (output)
        total_saved += output->descriptor.size_bytes;
    }
    if (!add_node->output_tensor_ids.empty())
    {
      TensorImpl *output = GetTensor(graph, add_node->output_tensor_ids[0]);
      if (output)
        total_saved += output->descriptor.size_bytes;
    }

    out_group->estimated_memory_saved_bytes = total_saved;
    out_group->estimated_speedup = 1.8; // 80% speedup estimate

    return true;
  }

} // namespace og_internal

/* ========================================================================== */
/* Public Pattern Detection API                                               */
/* ========================================================================== */

int og_detect_fusion_patterns(og_graph_t graph,
                               og_fusion_group_t *out_groups,
                               size_t max_groups,
                               size_t *out_num_groups)
{
  if (!graph || !out_groups || !out_num_groups)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  if (!impl->is_finalized)
  {
    return OG_ERR_INVALID_GRAPH;
  }

  size_t count = 0;
  for (auto &group_pair : impl->fusion_groups)
  {
    if (count >= max_groups)
      break;
    out_groups[count++] = group_pair.second->descriptor;
  }

  *out_num_groups = count;
  return OG_OK;
}

int og_get_fusion_info(og_graph_t graph,
                       uint64_t node_id,
                       og_fusion_pattern_t *out_pattern,
                       og_fusion_group_t *out_group)
{
  if (!graph || !out_pattern)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  // Find if this node is part of any fusion group
  for (auto &group_pair : impl->fusion_groups)
  {
    FusionGroupImpl *group = group_pair.second.get();
    for (size_t i = 0; i < group->descriptor.num_nodes; ++i)
    {
      if (group->descriptor.node_ids[i] == node_id)
      {
        *out_pattern = group->descriptor.pattern;
        if (out_group)
        {
          *out_group = group->descriptor;
        }
        return OG_OK;
      }
    }
  }

  *out_pattern = OG_PATTERN_NONE;
  return OG_OK;
}

int og_get_execution_order(og_graph_t graph,
                           uint64_t *out_order,
                           size_t max_nodes,
                           size_t *out_num_nodes)
{
  if (!graph || !out_order || !out_num_nodes)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  if (!impl->is_finalized)
  {
    return OG_ERR_INVALID_GRAPH;
  }

  size_t count = 0;
  for (uint64_t node_id : impl->topological_order)
  {
    if (count >= max_nodes)
      break;
    out_order[count++] = node_id;
  }

  *out_num_nodes = count;
  return OG_OK;
}