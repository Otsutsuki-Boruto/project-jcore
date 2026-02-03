// advanced/operator_graph/src/operator_graph_utils.cpp
/**
 * @file operator_graph_utils.cpp
 * @brief Utility functions - string conversions, statistics, graph export
 */

#include "operator_graph_internal.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace og_internal;

/* ========================================================================== */
/* String Conversion Functions                                                */
/* ========================================================================== */

const char *og_op_type_name(og_op_type_t op_type)
{
  switch (op_type)
  {
  case OG_OP_GEMM:
    return "GEMM";
  case OG_OP_BIAS_ADD:
    return "BiasAdd";
  case OG_OP_ELEMENTWISE_ADD:
    return "ElementwiseAdd";
  case OG_OP_ELEMENTWISE_MUL:
    return "ElementwiseMul";
  case OG_OP_RELU:
    return "ReLU";
  case OG_OP_RELU6:
    return "ReLU6";
  case OG_OP_TANH:
    return "Tanh";
  case OG_OP_SIGMOID:
    return "Sigmoid";
  case OG_OP_GELU:
    return "GELU";
  case OG_OP_SWISH:
    return "Swish";
  case OG_OP_LEAKY_RELU:
    return "LeakyReLU";
  case OG_OP_BATCH_NORM:
    return "BatchNorm";
  case OG_OP_LAYER_NORM:
    return "LayerNorm";
  case OG_OP_SOFTMAX:
    return "Softmax";
  case OG_OP_DROPOUT:
    return "Dropout";
  case OG_OP_CONV2D:
    return "Conv2D";
  case OG_OP_POOLING:
    return "Pooling";
  case OG_OP_CONCAT:
    return "Concat";
  case OG_OP_SPLIT:
    return "Split";
  case OG_OP_RESHAPE:
    return "Reshape";
  case OG_OP_TRANSPOSE:
    return "Transpose";
  case OG_OP_CUSTOM:
    return "Custom";
  default:
    return "Unknown";
  }
}

const char *og_pattern_name(og_fusion_pattern_t pattern)
{
  switch (pattern)
  {
  case OG_PATTERN_NONE:
    return "None";
  case OG_PATTERN_GEMM_BIAS:
    return "GEMM+Bias";
  case OG_PATTERN_GEMM_BIAS_ACTIVATION:
    return "GEMM+Bias+Activation";
  case OG_PATTERN_GEMM_ADD:
    return "GEMM+Add";
  case OG_PATTERN_GEMM_RESIDUAL_ACTIVATION:
    return "GEMM+Bias+Residual+Activation";
  case OG_PATTERN_CONV_BIAS_ACTIVATION:
    return "Conv+Bias+Activation";
  case OG_PATTERN_LINEAR_CHAIN:
    return "LinearChain";
  case OG_PATTERN_BATCH_GEMM:
    return "BatchGEMM";
  case OG_PATTERN_ACTIVATION_CHAIN:
    return "ActivationChain";
  default:
    return "Unknown";
  }
}

const char *og_strerror(int error)
{
  switch (error)
  {
  case OG_OK:
    return "Success";
  case OG_ERR_NOT_INITIALIZED:
    return "Operator Graph Runtime not initialized";
  case OG_ERR_INVALID_ARG:
    return "Invalid argument";
  case OG_ERR_NO_MEMORY:
    return "Out of memory";
  case OG_ERR_INTERNAL:
    return "Internal error";
  case OG_ERR_CYCLE_DETECTED:
    return "Cycle detected in graph";
  case OG_ERR_NODE_NOT_FOUND:
    return "Node not found";
  case OG_ERR_INVALID_GRAPH:
    return "Invalid graph state";
  case OG_ERR_FUSION_FAILED:
    return "Fusion execution failed";
  case OG_ERR_UNSUPPORTED:
    return "Unsupported operation or feature";
  default:
    return "Unknown error";
  }
}

/* ========================================================================== */
/* Graph Statistics                                                           */
/* ========================================================================== */

int og_get_graph_stats(og_graph_t graph, og_graph_stats_t *stats)
{
  if (!graph || !stats)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  *stats = impl->stats;

  // Find bottleneck operation (slowest node)
  double max_time = 0.0;
  const char *bottleneck = "N/A";

  for (auto &node_pair : impl->nodes)
  {
    NodeImpl *node = node_pair.second.get();
    if (node->execution_time_ms > max_time)
    {
      max_time = node->execution_time_ms;
      bottleneck = og_op_type_name(node->descriptor.type);
    }
  }

  stats->bottleneck_op = bottleneck;

  return OG_OK;
}

/* ========================================================================== */
/* Memory Estimation                                                          */
/* ========================================================================== */

namespace og_internal
{

  size_t EstimatePeakMemory(GraphImpl *graph)
  {
    size_t peak_memory = 0;
    size_t current_memory = 0;

    // Simulate execution and track memory usage
    std::unordered_set<uint64_t> live_tensors;

    for (uint64_t node_id : graph->topological_order)
    {
      NodeImpl *node = GetNode(graph, node_id);
      if (!node)
        continue;

      // Add input tensors to live set
      for (uint64_t tid : node->input_tensor_ids)
      {
        if (live_tensors.find(tid) == live_tensors.end())
        {
          TensorImpl *tensor = GetTensor(graph, tid);
          if (tensor)
          {
            current_memory += tensor->descriptor.size_bytes;
            live_tensors.insert(tid);
          }
        }
      }

      // Add output tensors to live set
      for (uint64_t tid : node->output_tensor_ids)
      {
        if (live_tensors.find(tid) == live_tensors.end())
        {
          TensorImpl *tensor = GetTensor(graph, tid);
          if (tensor)
          {
            current_memory += tensor->descriptor.size_bytes;
            live_tensors.insert(tid);
          }
        }
      }

      peak_memory = std::max(peak_memory, current_memory);

      // Remove dead tensors (simplified: assume tensor dies after last use)
      // In real implementation, would need liveness analysis
    }

    return peak_memory;
  }

} // namespace og_internal

int og_estimate_memory_usage(og_graph_t graph, size_t *out_peak_memory_bytes)
{
  if (!graph || !out_peak_memory_bytes)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  *out_peak_memory_bytes = EstimatePeakMemory(impl);
  return OG_OK;
}

/* ========================================================================== */
/* Graph Optimization                                                         */
/* ========================================================================== */

int og_optimize_graph(og_graph_t graph)
{
  if (!graph)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  if (!impl->is_finalized)
  {
    return OG_ERR_INVALID_GRAPH;
  }

  if (impl->is_optimized)
  {
    return OG_OK; // Already optimized
  }

  if (g_og_state.config.verbose)
  {
    std::cout << "[OG] Optimizing graph..." << std::endl;
  }

  // Optimization passes:
  // 1. Dead code elimination (remove unused nodes)
  // 2. Constant folding (evaluate constant expressions)
  // 3. Memory layout optimization
  // 4. Additional fusion opportunities

  // For now, mark as optimized
  impl->is_optimized = true;

  if (g_og_state.config.verbose)
  {
    std::cout << "[OG] Graph optimization complete" << std::endl;
  }

  return OG_OK;
}

/* ========================================================================== */
/* Graph Printing & Debugging                                                 */
/* ========================================================================== */

void og_print_graph(og_graph_t graph, int verbose)
{
  if (!graph)
  {
    std::cout << "[OG] NULL graph" << std::endl;
    return;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  std::cout << "\n=== Operator Graph ===" << std::endl;
  std::cout << "Nodes: " << impl->nodes.size() << std::endl;
  std::cout << "Tensors: " << impl->tensors.size() << std::endl;
  std::cout << "Edges: " << impl->stats.total_edges << std::endl;
  std::cout << "Fusion Groups: " << impl->fusion_groups.size() << std::endl;
  std::cout << "Finalized: " << (impl->is_finalized ? "Yes" : "No") << std::endl;

  if (verbose)
  {
    std::cout << "\n--- Nodes ---" << std::endl;
    for (auto &node_pair : impl->nodes)
    {
      NodeImpl *node = node_pair.second.get();
      std::cout << "  Node " << node->descriptor.id << ": "
                << og_op_type_name(node->descriptor.type)
                << " (inputs=" << node->input_tensor_ids.size()
                << ", outputs=" << node->output_tensor_ids.size()
                << ", successors=" << node->successor_nodes.size() << ")" << std::endl;

      if (node->execution_time_ms > 0.0)
      {
        std::cout << "    Execution time: " << std::fixed << std::setprecision(3)
                  << node->execution_time_ms << " ms" << std::endl;
      }
    }

    std::cout << "\n--- Tensors ---" << std::endl;
    for (auto &tensor_pair : impl->tensors)
    {
      TensorImpl *tensor = tensor_pair.second.get();
      std::cout << "  Tensor " << tensor->descriptor.id << ": ";
      std::cout << "shape=[";
      for (size_t i = 0; i < tensor->descriptor.ndim; ++i)
      {
        std::cout << tensor->descriptor.shape[i];
        if (i < tensor->descriptor.ndim - 1)
          std::cout << "x";
      }
      std::cout << "], elements=" << tensor->descriptor.total_elements
                << ", bytes=" << tensor->descriptor.size_bytes << std::endl;
    }

    if (!impl->fusion_groups.empty())
    {
      std::cout << "\n--- Fusion Groups ---" << std::endl;
      for (auto &group_pair : impl->fusion_groups)
      {
        FusionGroupImpl *group = group_pair.second.get();
        std::cout << "  Group " << group->descriptor.group_id << ": "
                  << og_pattern_name(group->descriptor.pattern)
                  << " (nodes=" << group->descriptor.num_nodes
                  << ", memory_saved=" << (group->descriptor.estimated_memory_saved_bytes / 1024.0) << " KB"
                  << ", speedup=" << group->descriptor.estimated_speedup << "x)" << std::endl;
      }
    }
  }

  std::cout << "=====================\n"
            << std::endl;
}

/* ========================================================================== */
/* Graph Export (DOT format)                                                  */
/* ========================================================================== */

int og_export_graph_dot(og_graph_t graph, const char *filename)
{
  if (!graph || !filename)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  std::ofstream out(filename);
  if (!out.is_open())
  {
    if (g_og_state.config.verbose)
    {
      std::cerr << "[OG] Failed to open file: " << filename << std::endl;
    }
    return OG_ERR_INTERNAL;
  }

  out << "digraph OperatorGraph {" << std::endl;
  out << "  rankdir=TB;" << std::endl;
  out << "  node [shape=box, style=rounded];" << std::endl;

  // Write nodes
  for (auto &node_pair : impl->nodes)
  {
    NodeImpl *node = node_pair.second.get();
    std::string label = og_op_type_name(node->descriptor.type);

    // Add shape info to label
    if (!node->output_tensor_ids.empty())
    {
      TensorImpl *out_tensor = GetTensor(impl, node->output_tensor_ids[0]);
      if (out_tensor && out_tensor->descriptor.ndim >= 2)
      {
        std::ostringstream shape_str;
        shape_str << "\\n[" << out_tensor->descriptor.shape[0]
                  << "x" << out_tensor->descriptor.shape[1] << "]";
        label += shape_str.str();
      }
    }

    // Color by fusion status
    std::string color = "lightgray";
    for (auto &group_pair : impl->fusion_groups)
    {
      FusionGroupImpl *group = group_pair.second.get();
      for (size_t i = 0; i < group->descriptor.num_nodes; ++i)
      {
        if (group->descriptor.node_ids[i] == node->descriptor.id)
        {
          color = "lightblue";
          break;
        }
      }
    }

    out << "  node" << node->descriptor.id
        << " [label=\"" << label << "\", fillcolor=" << color << ", style=filled];" << std::endl;
  }

  // Write edges
  for (auto &node_pair : impl->nodes)
  {
    NodeImpl *node = node_pair.second.get();
    for (uint64_t succ_id : node->successor_nodes)
    {
      out << "  node" << node->descriptor.id << " -> node" << succ_id << ";" << std::endl;
    }
  }

  // Add legend for fusion groups
  if (!impl->fusion_groups.empty())
  {
    out << "\n  // Fusion Groups Legend" << std::endl;
    out << "  subgraph cluster_legend {" << std::endl;
    out << "    label=\"Fusion Groups\";" << std::endl;
    int idx = 0;
    for (auto &group_pair : impl->fusion_groups)
    {
      FusionGroupImpl *group = group_pair.second.get();
      out << "    legend" << idx++ << " [label=\""
          << og_pattern_name(group->descriptor.pattern)
          << " (Group " << group->descriptor.group_id << ")\", shape=note];" << std::endl;
    }
    out << "  }" << std::endl;
  }

  out << "}" << std::endl;
  out.close();

  if (g_og_state.config.verbose)
  {
    std::cout << "[OG] Exported graph to: " << filename << std::endl;
  }

  return OG_OK;
}