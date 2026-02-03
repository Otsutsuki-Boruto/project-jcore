// advanced/operator_graph/src/operator_graph_construction.cpp
/**
 * @file operator_graph_construction.cpp
 * @brief Graph construction operations - adding nodes, tensors, edges
 */

#include "operator_graph_internal.h"
#include <iostream>

using namespace og_internal;

/* ========================================================================== */
/* Tensor Management                                                          */
/* ========================================================================== */

int og_add_tensor(og_graph_t graph, const og_tensor_t *tensor,
                  uint64_t *out_tensor_id) {

  if (!graph || !tensor || !out_tensor_id) {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  if (impl->is_finalized) {
    if (g_og_state.config.verbose) {
      std::cerr << "[OG] Cannot add tensor to finalized graph" << std::endl;
    }
    return OG_ERR_INVALID_GRAPH;
  }

  // Validate tensor descriptor
  if (tensor->ndim == 0 || tensor->ndim > OG_MAX_TENSOR_DIMS) {
    return OG_ERR_INVALID_ARG;
  }

  try {
    // Create tensor implementation
    auto tensor_impl = std::make_unique<TensorImpl>(*tensor);

    // Assign unique ID if not provided
    if (tensor->id == 0) {
      tensor_impl->descriptor.id =
          impl->next_tensor_id.fetch_add(1, std::memory_order_relaxed);
    }

    // Calculate total elements and size
    size_t total = 1;
    for (size_t i = 0; i < tensor->ndim; ++i) {
      if (tensor->shape[i] == 0) {
        return OG_ERR_INVALID_ARG;
      }
      total *= tensor->shape[i];
    }
    tensor_impl->descriptor.total_elements = total;
    tensor_impl->descriptor.size_bytes =
        total * sizeof(float); // Assuming float32

    // Calculate row-major strides if not provided
    if (tensor->strides[0] == 0) {
      // Row-major (C-style) strides
      size_t stride = 1;
      for (int i = tensor->ndim - 1; i >= 0; --i) {
        tensor_impl->descriptor.strides[i] = stride;
        stride *= tensor->shape[i];
      }
    }

    // If data is not provided, allocate it
    if (!tensor_impl->descriptor.data && !tensor->is_constant) {
      void *data = AllocateTensorData(tensor_impl->descriptor.size_bytes);
      if (!data) {
        return OG_ERR_NO_MEMORY;
      }
      tensor_impl->descriptor.data = data;
      tensor_impl->allocated_data = data;
      tensor_impl->allocated_size = tensor_impl->descriptor.size_bytes;
    } else if (tensor_impl->descriptor.data) {
      // User provided data - verify it's contiguous row-major
      // For now, we trust the user's data is properly laid out
      // In production, we might want to copy to ensure contiguity
    }

    *out_tensor_id = tensor_impl->descriptor.id;

    // Insert into graph
    impl->tensors[tensor_impl->descriptor.id] = std::move(tensor_impl);
    impl->stats.total_tensors++;

    if (g_og_state.config.verbose) {
      std::cout << "[OG] Added tensor ID=" << *out_tensor_id << ", shape=[";
      for (size_t i = 0; i < tensor->ndim; ++i) {
        std::cout << tensor->shape[i];
        if (i < tensor->ndim - 1)
          std::cout << "x";
      }
      std::cout << "], elements=" << total
                << ", bytes=" << tensor_impl->descriptor.size_bytes
                << std::endl;
    }

    return OG_OK;
  } catch (const std::bad_alloc &) {
    return OG_ERR_NO_MEMORY;
  } catch (...) {
    return OG_ERR_INTERNAL;
  }
}

int og_get_tensor_info(og_graph_t graph, uint64_t tensor_id,
                       og_tensor_t *out_tensor) {
  if (!graph || !out_tensor) {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  TensorImpl *tensor = GetTensor(impl, tensor_id);
  if (!tensor) {
    return OG_ERR_NODE_NOT_FOUND;
  }

  *out_tensor = tensor->descriptor;
  return OG_OK;
}

/* ========================================================================== */
/* Node Management                                                            */
/* ========================================================================== */

int og_add_node(og_graph_t graph, const og_node_t *node,
                uint64_t *out_node_id) {

  if (!graph || !node || !out_node_id) {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  if (impl->is_finalized) {
    if (g_og_state.config.verbose) {
      std::cerr << "[OG] Cannot add node to finalized graph" << std::endl;
    }
    return OG_ERR_INVALID_GRAPH;
  }

  // Validate node
  if (node->num_inputs == 0 && node->type != OG_OP_CUSTOM) {
    return OG_ERR_INVALID_ARG;
  }

  if (node->num_inputs > OG_MAX_INPUTS || node->num_outputs > OG_MAX_OUTPUTS) {
    return OG_ERR_INVALID_ARG;
  }

  try {
    // Create node implementation
    auto node_impl = std::make_unique<NodeImpl>(*node);

    // Assign unique ID if not provided
    if (node->id == 0) {
      node_impl->descriptor.id =
          impl->next_node_id.fetch_add(1, std::memory_order_relaxed);
    }

    // Validate all input tensors exist
    for (size_t i = 0; i < node->num_inputs; ++i) {
      uint64_t tid = node->input_ids[i];
      if (!GetTensor(impl, tid)) {
        if (g_og_state.config.verbose) {
          std::cerr << "[OG] Input tensor ID=" << tid << " not found"
                    << std::endl;
        }
        return OG_ERR_INVALID_ARG;
      }
      node_impl->input_tensor_ids.push_back(tid);
    }

    for (size_t i = 0; i < node->num_outputs; ++i) {
      uint64_t tid = node->output_ids[i];
      if (!GetTensor(impl, tid)) {
        if (g_og_state.config.verbose) {
          std::cerr << "[OG] Output tensor ID=" << tid << " not found"
                    << std::endl;
        }
        return OG_ERR_INVALID_ARG;
      }
      node_impl->output_tensor_ids.push_back(tid);
    }

    *out_node_id = node_impl->descriptor.id;

    // Insert into graph
    impl->nodes[node_impl->descriptor.id] = std::move(node_impl);
    impl->stats.total_nodes++;


    if (g_og_state.config.verbose) {
      std::cout << "[OG] Added node ID=" << *out_node_id
                << ", type=" << og_op_type_name(node->type)
                << ", inputs=" << node->num_inputs
                << ", outputs=" << node->num_outputs << std::endl;
    }

    return OG_OK;
  } catch (const std::bad_alloc &) {
    return OG_ERR_NO_MEMORY;
  } catch (...) {
    return OG_ERR_INTERNAL;
  }
}

int og_get_node_info(og_graph_t graph, uint64_t node_id, og_node_t *out_node) {
  if (!graph || !out_node) {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  NodeImpl *node = GetNode(impl, node_id);
  if (!node) {
    return OG_ERR_NODE_NOT_FOUND;
  }

  *out_node = node->descriptor;
  return OG_OK;
}

/* ========================================================================== */
/* Edge Management                                                            */
/* ========================================================================== */

int og_add_edge(og_graph_t graph, uint64_t from_node, uint64_t to_node) {
  if (!graph) {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  if (impl->is_finalized) {
    if (g_og_state.config.verbose) {
      std::cerr << "[OG] Cannot add edge to finalized graph" << std::endl;
    }
    return OG_ERR_INVALID_GRAPH;
  }

  NodeImpl *from = GetNode(impl, from_node);
  NodeImpl *to = GetNode(impl, to_node);

  if (!from || !to) {
    return OG_ERR_NODE_NOT_FOUND;
  }

  // Check if edge already exists
  for (uint64_t succ : from->successor_nodes) {
    if (succ == to_node) {
      if (g_og_state.config.verbose) {
        std::cout << "[OG] Edge " << from_node << " -> " << to_node
                  << " already exists, skipping" << std::endl;
      }
      return OG_OK;  // Edge already exists, not an error
    }
  }

  // Add edge: from -> to
  from->successor_nodes.push_back(to_node);
  to->predecessor_nodes.push_back(from_node);
  to->in_degree++;

  impl->stats.total_edges++;

  if (g_og_state.config.verbose) {
    std::cout << "[OG] Added edge: " << from_node << " -> " << to_node
              << std::endl;
  }

  return OG_OK;
}

/* ========================================================================== */
/* Graph Finalization                                                         */
/* ========================================================================== */

int og_finalize_graph(og_graph_t graph) {
  if (!graph) {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  if (impl->is_finalized) {
    return OG_OK; // Already finalized
  }

  if (g_og_state.config.verbose) {
    std::cout << "[OG] Finalizing graph with " << impl->nodes.size()
              << " nodes, " << impl->tensors.size() << " tensors" << std::endl;
  }

  // Step 1: Build adjacency lists
  if (g_og_state.config.verbose) {
    std::cout << "[OG] Step 1: Building adjacency lists..." << std::endl;
  }
  BuildAdjacencyLists(impl);

  // Step 2: Validate graph structure
  if (g_og_state.config.verbose) {
    std::cout << "[OG] Step 2: Validating tensor references..." << std::endl;
  }
  int ret = ValidateTensorReferences(impl);
  if (ret != OG_OK) {
    if (g_og_state.config.verbose) {
      std::cerr << "[OG] ERROR: Tensor reference validation failed: "
                << og_strerror(ret) << std::endl;
    }
    return ret;
  }

  // Step 3: Check for cycles
  if (g_og_state.config.verbose) {
    std::cout << "[OG] Step 3: Checking for cycles..." << std::endl;
  }

  if (HasCycle(impl)) {
    if (g_og_state.config.verbose) {
      std::cerr << "[OG] ERROR: Cycle detected in graph" << std::endl;
    }
    return OG_ERR_CYCLE_DETECTED;
  }

  // Step 4: Perform topological sort
  if (g_og_state.config.verbose) {
    std::cout << "[OG] Step 4: Computing topological order..." << std::endl;
  }
  ret = TopologicalSort(impl);
  if (ret != OG_OK) {
    if (g_og_state.config.verbose) {
      std::cerr << "[OG] ERROR: Topological sort failed: " << og_strerror(ret)
                << std::endl;
    }
    return ret;
  }

  // Step 5: Detect fusion patterns if enabled
  if (g_og_state.config.enable_pattern_matching) {
    if (g_og_state.config.verbose) {
      std::cout << "[OG] Step 5: Detecting fusion patterns..." << std::endl;
    }
    ret = AnalyzeAndDetectPatterns(impl);
    if (ret != OG_OK) {
      if (g_og_state.config.verbose) {
        std::cerr << "[OG] WARNING: Pattern detection failed: "
                  << og_strerror(ret) << std::endl;
        std::cerr << "[OG] Continuing without fusion optimization" << std::endl;
      }
      // Don't fail finalization if pattern detection fails
      ret = OG_OK;
    }
  }

  impl->is_finalized = true;

  if (g_og_state.config.verbose) {
    std::cout << "[OG] Graph finalized successfully" << std::endl;
    std::cout << "[OG]   Topological order: " << impl->topological_order.size()
              << " nodes" << std::endl;
    std::cout << "[OG]   Fusion groups: " << impl->fusion_groups.size()
              << std::endl;
  }

  return OG_OK;
}

/* ========================================================================== */
/* Graph Validation                                                           */
/* ========================================================================== */

int og_validate_graph(og_graph_t graph) {
  if (!graph) {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  // Check for cycles
  if (HasCycle(impl)) {
    return OG_ERR_CYCLE_DETECTED;
  }

  // Validate tensor references
  int ret = ValidateTensorReferences(impl);
  if (ret != OG_OK) {
    return ret;
  }

  return OG_OK;
}

/* ========================================================================== */
/* Helper Functions                                                           */
/* ========================================================================== */

namespace og_internal {

void BuildAdjacencyLists(GraphImpl *graph) {

  // Clear existing adjacency info
  for (auto &node_pair : graph->nodes) {
    NodeImpl *node = node_pair.second.get();
    node->successor_nodes.clear();
    node->predecessor_nodes.clear();
    node->in_degree = 0;
  }

  // Build adjacency lists by analyzing tensor data flow
  for (auto &node_pair : graph->nodes) {
    NodeImpl *node = node_pair.second.get();

    // For each output tensor of this node
    for (uint64_t output_tid : node->output_tensor_ids) {
      // Find all nodes that use this tensor as input
      for (auto &other_pair : graph->nodes) {
        if (other_pair.first == node->descriptor.id)
          continue; // Skip self

        NodeImpl *other = other_pair.second.get();

        // Check if 'other' uses this output as input
        for (uint64_t input_tid : other->input_tensor_ids) {
          if (input_tid == output_tid) {
            // Add edge: node -> other
            bool already_exists = false;
            for (uint64_t succ : node->successor_nodes) {
              if (succ == other->descriptor.id) {
                already_exists = true;
                break;
              }
            }

            if (!already_exists) {
              node->successor_nodes.push_back(other->descriptor.id);
              other->predecessor_nodes.push_back(node->descriptor.id);
              other->in_degree++;

              if (g_og_state.config.verbose) {
                std::cout << "[OG] Edge: Node" << node->descriptor.id
                          << " -> Node" << other->descriptor.id
                          << " (via tensor " << output_tid << ")" << std::endl;
              }
            }
            break; // Only add edge once per node pair
          }
        }
      }
    }
  }

  if (g_og_state.config.verbose) {
    std::cout << "[OG] Adjacency lists built:" << std::endl;
    for (auto &node_pair : graph->nodes) {
      NodeImpl *node = node_pair.second.get();
      std::cout << "[OG]   Node" << node->descriptor.id
                << ": in_degree=" << node->in_degree
                << ", successors=" << node->successor_nodes.size()
                << ", predecessors=" << node->predecessor_nodes.size()
                << std::endl;
    }
  }
}

int ValidateTensorReferences(GraphImpl *graph) {
  for (auto &node_pair : graph->nodes) {
    NodeImpl *node = node_pair.second.get();

    // Check all input tensors exist
    for (uint64_t tid : node->input_tensor_ids) {
      if (!GetTensor(graph, tid)) {
        if (g_og_state.config.verbose) {
          std::cerr << "[OG] ERROR: Node " << node->descriptor.id
                    << " references non-existent input tensor " << tid
                    << std::endl;
        }
        return OG_ERR_INVALID_ARG;
      }
    }

    // Check all output tensors exist
    for (uint64_t tid : node->output_tensor_ids) {
      if (!GetTensor(graph, tid)) {
        if (g_og_state.config.verbose) {
          std::cerr << "[OG] ERROR: Node " << node->descriptor.id
                    << " references non-existent output tensor " << tid
                    << std::endl;
        }
        return OG_ERR_INVALID_ARG;
      }
    }

    // NOTE: In-place operations (same tensor as input and output) are ALLOWED
    // This is common for operations like Bias, Activation, etc.
    // Example: Bias node reads from temp_tensor and writes back to temp_tensor
  }

  return OG_OK;
}

} // namespace og_internal