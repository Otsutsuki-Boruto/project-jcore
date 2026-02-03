// advanced/operator_graph/src/operator_graph_execution.cpp
/**
 * @file operator_graph_execution.cpp
 * @brief Graph execution engine - execute nodes and fusion groups
 */

#include "operator_graph_internal.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "ffm_prefetch.h"

#include "kernel_fusion_engine_internal.h"

using namespace og_internal;

/* ========================================================================== */
/* Graph Execution - Main Entry Point                                        */
/* ========================================================================== */

int og_execute_graph(og_graph_t graph, og_graph_stats_t *stats)
{
  if (!graph)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  if (!impl->is_finalized)
  {
    if (g_og_state.config.verbose)
    {
      std::cerr << "[OG] Cannot execute non-finalized graph" << std::endl;
    }
    return OG_ERR_INVALID_GRAPH;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Reset execution state
  for (auto &node_pair : impl->nodes)
  {
    node_pair.second->is_executed = false;
    node_pair.second->execution_time_ms = 0.0;
  }

  size_t fused_ops = 0;
  size_t memory_saved = 0;

  // Execute fusion groups first
  if (g_og_state.config.enable_fusion && !impl->fusion_groups.empty())
  {
    if (g_og_state.config.verbose)
    {
      std::cout << "[OG] Executing " << impl->fusion_groups.size() << " fusion groups" << std::endl;
    }

    for (auto &group_pair : impl->fusion_groups)
    {
      FusionGroupImpl *group = group_pair.second.get();
      if (group->is_executed)
        continue;

      if (g_og_state.config.verbose)
      {
        std::cout << "[OG] Attempting to execute fusion group " << group->descriptor.group_id
                  << " (pattern=" << og_pattern_name(group->descriptor.pattern) << ")" << std::endl;
      }

      int ret = ExecuteFusionGroupViaKFE(impl, group);
      if (ret != OG_OK)
      {
        if (g_og_state.config.verbose)
        {
          std::cerr << "[OG] Fusion group " << group->descriptor.group_id
                    << " execution failed: " << og_strerror(ret)
                    << " - falling back to individual execution" << std::endl;
        }
        // Fall back to individual execution
        continue;
      }

      // Mark nodes in fusion group as executed
      for (size_t i = 0; i < group->descriptor.num_nodes; ++i)
      {
        uint64_t node_id = group->descriptor.node_ids[i];
        NodeImpl *node = GetNode(impl, node_id);
        if (node)
        {
          node->is_executed = true;
        }
      }

      group->is_executed = true;
      fused_ops += group->descriptor.num_nodes;
      memory_saved += group->descriptor.estimated_memory_saved_bytes;

      g_og_state.total_fusion_groups_executed.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Execute remaining nodes in topological order
  for (uint64_t node_id : impl->topological_order)
  {
    NodeImpl *node = GetNode(impl, node_id);
    if (!node || node->is_executed)
      continue;

    auto node_start = std::chrono::high_resolution_clock::now();

    int ret = OG_OK;
    switch (node->descriptor.type)
    {
    case OG_OP_GEMM:
      ret = ExecuteGemmNode(impl, node);
      break;

    case OG_OP_BIAS_ADD:
      ret = ExecuteBiasAddNode(impl, node);
      break;

    case OG_OP_RELU:
    case OG_OP_RELU6:
    case OG_OP_TANH:
    case OG_OP_SIGMOID:
    case OG_OP_GELU:
    case OG_OP_SWISH:
    case OG_OP_LEAKY_RELU:
      ret = ExecuteActivationNode(impl, node);
      break;

    case OG_OP_ELEMENTWISE_ADD:
    case OG_OP_ELEMENTWISE_MUL:
      ret = ExecuteElementwiseNode(impl, node);
      break;

    default:
      if (g_og_state.config.verbose)
      {
        std::cerr << "[OG] Unsupported operation type: " << og_op_type_name(node->descriptor.type) << std::endl;
      }
      ret = OG_ERR_UNSUPPORTED;
      break;
    }

    auto node_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> node_elapsed = node_end - node_start;
    node->execution_time_ms = node_elapsed.count();

    if (ret != OG_OK)
    {
      if (g_og_state.config.verbose)
      {
        std::cerr << "[OG] Node execution failed: " << og_strerror(ret) << std::endl;
      }
      return ret;
    }

    node->is_executed = true;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

  // Update statistics
  impl->stats.total_ops_fused = fused_ops;
  impl->stats.total_memory_saved_bytes = memory_saved;
  impl->stats.total_execution_time_ms = elapsed.count();

  if (fused_ops > 0)
  {
    impl->stats.avg_fusion_speedup = static_cast<double>(fused_ops) / impl->fusion_groups.size();
  }

  if (stats)
  {
    *stats = impl->stats;
  }

  g_og_state.total_graphs_executed.fetch_add(1, std::memory_order_relaxed);

  if (g_og_state.config.verbose)
  {
    std::cout << "[OG] Graph execution complete in " << elapsed.count() << " ms" << std::endl;
    std::cout << "[OG]   Fusion groups executed: " << impl->fusion_groups.size() << std::endl;
    std::cout << "[OG]   Operations fused: " << fused_ops << std::endl;
    std::cout << "[OG]   Memory saved: " << (memory_saved / 1024.0 / 1024.0) << " MB" << std::endl;
  }

  return OG_OK;
}

/* ========================================================================== */
/* Execute Individual Node Types                                              */
/* ========================================================================== */

namespace og_internal
{

  int ExecuteGemmNode(GraphImpl *graph, NodeImpl *node)
  {

    // Get input tensors (A, B)
    if (node->input_tensor_ids.size() < 2)
    {
      return OG_ERR_INVALID_ARG;
    }

    TensorImpl *A = GetTensor(graph, node->input_tensor_ids[0]);
    TensorImpl *B = GetTensor(graph, node->input_tensor_ids[1]);

    if (!A || !B)
    {
      return OG_ERR_INVALID_ARG;
    }

    // Get output tensor
    if (node->output_tensor_ids.empty())
    {
      return OG_ERR_INVALID_ARG;
    }

    TensorImpl *C = GetTensor(graph, node->output_tensor_ids[0]);
    if (!C)
    {
      return OG_ERR_INVALID_ARG;
    }

    // Prefetch input tensors
    ffm_prefetch_block_read_T0(A->descriptor.data, A->descriptor.total_elements * sizeof(float));
    ffm_prefetch_block_read_T0(B->descriptor.data, B->descriptor.total_elements * sizeof(float));
    ffm_prefetch_block_write_T0(C->descriptor.data, C->descriptor.total_elements * sizeof(float));

    // Extract GEMM parameters from tensor shapes
    // A is [M x K], B is [K x N], C is [M x N]
    size_t M = A->descriptor.shape[0];
    size_t K = A->descriptor.shape[1];
    size_t N = B->descriptor.shape[1];

    // Verify dimensions match
    if (B->descriptor.shape[0] != K)
    {
      if (g_og_state.config.verbose)
      {
        std::cerr << "[OG] GEMM dimension mismatch: A.K=" << K
                  << " != B.M=" << B->descriptor.shape[0] << std::endl;
      }
      return OG_ERR_INVALID_ARG;
    }

    if (C->descriptor.shape[0] != M || C->descriptor.shape[1] != N)
    {
      if (g_og_state.config.verbose)
      {
        std::cerr << "[OG] GEMM output dimension mismatch" << std::endl;
      }
      return OG_ERR_INVALID_ARG;
    }

    // Get alpha from attributes (default 1.0)
    float alpha = (node->descriptor.num_attributes > 0) ? node->descriptor.attributes[0] : 1.0f;

    float *bias_ptr = nullptr;
    std::vector<float> zero_bias;

    if (node->input_tensor_ids.size() >= 3)
    {
      TensorImpl *bias_tensor = GetTensor(graph, node->input_tensor_ids[2]);
      if (bias_tensor && bias_tensor->descriptor.total_elements == N)
      {
        bias_ptr = static_cast<float *>(bias_tensor->descriptor.data);
      }
      else
      {
        // Fallback: create zero bias of correct length
        zero_bias.resize(N, 0.0f);
        bias_ptr = zero_bias.data();
      }
    }
    else
    {
      // No bias tensor, create zero bias
      zero_bias.resize(N, 0.0f);
      bias_ptr = zero_bias.data();
    }

    kfe_perf_stats_t kfe_stats = {};
    int ret = kfe_sgemm_bias(
        KFE_LAYOUT_ROW_MAJOR,
        KFE_NO_TRANS, KFE_NO_TRANS,
        M, N, K,
        alpha,
        static_cast<const float *>(A->descriptor.data), K,
        static_cast<const float *>(B->descriptor.data), N,
        bias_ptr,
        static_cast<float *>(C->descriptor.data), N,
        &kfe_stats);

    if (ret != KFE_OK)
    {
      if (g_og_state.config.verbose)
        std::cerr << "[OG] KFE failed with code: " << ret << std::endl;
      return OG_ERR_INTERNAL;
    }

    return OG_OK;
  }

  int ExecuteBiasAddNode(GraphImpl *graph, NodeImpl *node)
  {
    // Input: matrix [M x N], bias vector [N]
    if (node->input_tensor_ids.size() < 2)
    {
      return OG_ERR_INVALID_ARG;
    }

    TensorImpl *input = GetTensor(graph, node->input_tensor_ids[0]);
    TensorImpl *bias = GetTensor(graph, node->input_tensor_ids[1]);

    if (!input || !bias)
    {
      return OG_ERR_INVALID_ARG;
    }

    if (node->output_tensor_ids.empty())
    {
      return OG_ERR_INVALID_ARG;
    }

    TensorImpl *output = GetTensor(graph, node->output_tensor_ids[0]);
    if (!output)
    {
      return OG_ERR_INVALID_ARG;
    }

    // Prefetch input tensors
    ffm_prefetch_block_read_T0(input->descriptor.data, input->descriptor.total_elements * sizeof(float));
    ffm_prefetch_block_read_T0(bias->descriptor.data, bias->descriptor.total_elements * sizeof(float));
    ffm_prefetch_block_write_T0(output->descriptor.data, output->descriptor.total_elements * sizeof(float));

    size_t M = input->descriptor.shape[0];
    size_t N = input->descriptor.shape[1];

    const float *in_data = static_cast<const float *>(input->descriptor.data);
    const float *bias_data = static_cast<const float *>(bias->descriptor.data);
    float *out_data = static_cast<float *>(output->descriptor.data);

    // Broadcast bias across rows
    // Copy input to output
    std::memcpy(out_data, in_data, M * N * sizeof(float));

    // Apply bias efficiently via KFE scalar path
    kfe_internal::add_bias_row_major(out_data, M, N, N, static_cast<const float *>(bias->descriptor.data));

    return OG_OK;
  }

  int ExecuteActivationNode(GraphImpl *graph, NodeImpl *node)
  {
    if (node->input_tensor_ids.empty())
    {
      return OG_ERR_INVALID_ARG;
    }

    TensorImpl *input = GetTensor(graph, node->input_tensor_ids[0]);
    if (!input)
    {
      return OG_ERR_INVALID_ARG;
    }

    if (node->output_tensor_ids.empty())
    {
      return OG_ERR_INVALID_ARG;
    }

    TensorImpl *output = GetTensor(graph, node->output_tensor_ids[0]);
    if (!output)
    {
      return OG_ERR_INVALID_ARG;
    }

    // Prefetch input/output tensors
    ffm_prefetch_block_read_T0(input->descriptor.data, input->descriptor.total_elements * sizeof(float));
    ffm_prefetch_block_write_T0(output->descriptor.data, output->descriptor.total_elements * sizeof(float));

    const float *in_data = static_cast<const float *>(input->descriptor.data);

    float *out_data = static_cast<float *>(output->descriptor.data);
    size_t n = input->descriptor.total_elements;

    // Apply activation function
    kfe_activation_t act = OpTypeToKFEActivation(node->descriptor.type);

    // Replace element-wise scalar loop with KFE vectorized activation
    kfe_internal::apply_activation_vectorized(out_data, n, act);

    return OG_OK;
  }

  int ExecuteElementwiseNode(GraphImpl *graph, NodeImpl *node)
  {
    if (node->input_tensor_ids.size() < 2)
      return OG_ERR_INVALID_ARG;

    TensorImpl *A = GetTensor(graph, node->input_tensor_ids[0]);
    TensorImpl *B = GetTensor(graph, node->input_tensor_ids[1]);
    if (!A || !B)
      return OG_ERR_INVALID_ARG;

    if (node->output_tensor_ids.empty())
      return OG_ERR_INVALID_ARG;

    TensorImpl *C = GetTensor(graph, node->output_tensor_ids[0]);
    if (!C)
      return OG_ERR_INVALID_ARG;

    // Prefetch input tensors
    ffm_prefetch_block_read_T0(A->descriptor.data, A->descriptor.total_elements * sizeof(float));
    ffm_prefetch_block_read_T0(B->descriptor.data, B->descriptor.total_elements * sizeof(float));
    ffm_prefetch_block_write_T0(C->descriptor.data, C->descriptor.total_elements * sizeof(float));

    if (A->descriptor.total_elements != B->descriptor.total_elements)
      return OG_ERR_INVALID_ARG;

    float *c_data = static_cast<float *>(C->descriptor.data);
    const float *a_data = static_cast<const float *>(A->descriptor.data);
    const float *b_data = static_cast<const float *>(B->descriptor.data);

    size_t M = A->descriptor.shape[0];
    size_t N = A->descriptor.shape[1];
    size_t lda = N;
    size_t ldb = N;
    size_t ldc = N;

    float beta = (node->descriptor.num_attributes > 0) ? node->descriptor.attributes[0] : 1.0f;

    if (node->descriptor.type == OG_OP_ELEMENTWISE_ADD)
    {
      kfe_internal::elementwise_add(c_data, M, N, ldc, b_data, ldb, beta);
    }
    else if (node->descriptor.type == OG_OP_ELEMENTWISE_MUL)
    {
      kfe_internal::elementwise_mul(c_data, M, N, ldc, b_data, ldb, beta);
    }
    else
    {
      return OG_ERR_UNSUPPORTED;
    }


    return OG_OK;
  }


  /* ========================================================================== */
  /* Execute Fusion Group via KFE                                               */
  /* ========================================================================== */

  int ExecuteFusionGroupViaKFE(GraphImpl *graph, FusionGroupImpl *group)
  {
    auto start_time = std::chrono::high_resolution_clock::now();

    int ret = OG_OK;

    switch (group->descriptor.pattern)
    {
    case OG_PATTERN_GEMM_BIAS:
    {
      // Execute GEMM + Bias fusion
      if (group->descriptor.num_nodes < 2)
        return OG_ERR_INTERNAL;

      NodeImpl *gemm = GetNode(graph, group->descriptor.node_ids[0]);
      NodeImpl *bias = GetNode(graph, group->descriptor.node_ids[1]);

      if (!gemm || !bias)
        return OG_ERR_INTERNAL;

      // Get tensors
      TensorImpl *A = GetTensor(graph, gemm->input_tensor_ids[0]);
      TensorImpl *B = GetTensor(graph, gemm->input_tensor_ids[1]);
      TensorImpl *bias_vec = GetTensor(graph, bias->input_tensor_ids[1]);
      TensorImpl *C = GetTensor(graph, bias->output_tensor_ids[0]);

      if (!A || !B || !bias_vec || !C)
        return OG_ERR_INTERNAL;

      // Prefetch input tensors
      ffm_prefetch_block_read_T0(A->descriptor.data, A->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_read_T0(B->descriptor.data, B->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_read_T0(bias_vec->descriptor.data, bias_vec->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_write_T0(C->descriptor.data, C->descriptor.total_elements * sizeof(float));

      size_t M = A->descriptor.shape[0];
      size_t K = A->descriptor.shape[1];
      size_t N = B->descriptor.shape[1];
      float alpha = (gemm->descriptor.num_attributes > 0) ? gemm->descriptor.attributes[0] : 1.0f;

      kfe_perf_stats_t kfe_stats = {};
      ret = kfe_sgemm_bias(
          KFE_LAYOUT_ROW_MAJOR,
          KFE_NO_TRANS, KFE_NO_TRANS,
          M, N, K, alpha,
          static_cast<const float *>(A->descriptor.data), K,
          static_cast<const float *>(B->descriptor.data), N,
          static_cast<const float *>(bias_vec->descriptor.data),
          static_cast<float *>(C->descriptor.data), N,
          &kfe_stats);

      break;
    }

    case OG_PATTERN_GEMM_BIAS_ACTIVATION:
    {
      // Execute GEMM + Bias + Activation fusion
      if (group->descriptor.num_nodes < 3)
        return OG_ERR_INTERNAL;

      NodeImpl *gemm = GetNode(graph, group->descriptor.node_ids[0]);
      NodeImpl *bias = GetNode(graph, group->descriptor.node_ids[1]);
      NodeImpl *act = GetNode(graph, group->descriptor.node_ids[2]);

      if (!gemm || !bias || !act)
        return OG_ERR_INTERNAL;

      // Get tensors
      TensorImpl *A = GetTensor(graph, gemm->input_tensor_ids[0]);
      TensorImpl *B = GetTensor(graph, gemm->input_tensor_ids[1]);
      TensorImpl *bias_vec = GetTensor(graph, bias->input_tensor_ids[1]);
      TensorImpl *C = GetTensor(graph, act->output_tensor_ids[0]);

      if (!A || !B || !bias_vec || !C)
        return OG_ERR_INTERNAL;

      // Prefetch input tensors
      ffm_prefetch_block_read_T0(A->descriptor.data, A->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_read_T0(B->descriptor.data, B->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_read_T0(bias_vec->descriptor.data, bias_vec->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_write_T0(C->descriptor.data, C->descriptor.total_elements * sizeof(float));

      size_t M = A->descriptor.shape[0];
      size_t K = A->descriptor.shape[1];
      size_t N = B->descriptor.shape[1];
      float alpha = (gemm->descriptor.num_attributes > 0) ? gemm->descriptor.attributes[0] : 1.0f;

      kfe_activation_t activation = OpTypeToKFEActivation(act->descriptor.type);

      kfe_perf_stats_t kfe_stats = {};
      ret = kfe_sgemm_bias_activation(
          KFE_LAYOUT_ROW_MAJOR,
          KFE_NO_TRANS, KFE_NO_TRANS,
          M, N, K, alpha,
          static_cast<const float *>(A->descriptor.data), K,
          static_cast<const float *>(B->descriptor.data), N,
          static_cast<const float *>(bias_vec->descriptor.data),
          activation,
          static_cast<float *>(C->descriptor.data), N,
          &kfe_stats);

      break;
    }

    case OG_PATTERN_GEMM_RESIDUAL_ACTIVATION:
    {
      // Execute GEMM + Bias + Residual + Activation fusion
      if (group->descriptor.num_nodes < 4)
        return OG_ERR_INTERNAL;

      NodeImpl *gemm = GetNode(graph, group->descriptor.node_ids[0]);
      NodeImpl *bias = GetNode(graph, group->descriptor.node_ids[1]);
      NodeImpl *add = GetNode(graph, group->descriptor.node_ids[2]);
      NodeImpl *act = GetNode(graph, group->descriptor.node_ids[3]);

      if (!gemm || !bias || !add || !act)
        return OG_ERR_INTERNAL;

      // Get tensors
      TensorImpl *A = GetTensor(graph, gemm->input_tensor_ids[0]);
      TensorImpl *B = GetTensor(graph, gemm->input_tensor_ids[1]);
      TensorImpl *bias_vec = GetTensor(graph, bias->input_tensor_ids[1]);
      TensorImpl *residual = GetTensor(graph, add->input_tensor_ids[1]);
      TensorImpl *C = GetTensor(graph, act->output_tensor_ids[0]);

      if (!A || !B || !bias_vec || !residual || !C)
        return OG_ERR_INTERNAL;

      // Prefetch input tensors
      ffm_prefetch_block_read_T0(A->descriptor.data, A->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_read_T0(B->descriptor.data, B->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_read_T0(bias_vec->descriptor.data, bias_vec->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_read_T0(residual->descriptor.data, residual->descriptor.total_elements * sizeof(float));
      ffm_prefetch_block_write_T0(C->descriptor.data, C->descriptor.total_elements * sizeof(float));

      size_t M = A->descriptor.shape[0];
      size_t K = A->descriptor.shape[1];
      size_t N = B->descriptor.shape[1];
      float alpha = (gemm->descriptor.num_attributes > 0) ? gemm->descriptor.attributes[0] : 1.0f;
      float beta = (add->descriptor.num_attributes > 0) ? add->descriptor.attributes[0] : 1.0f;

      kfe_activation_t activation = OpTypeToKFEActivation(act->descriptor.type);

      kfe_perf_stats_t kfe_stats = {};
      ret = kfe_sgemm_residual_activation(
          KFE_LAYOUT_ROW_MAJOR,
          KFE_NO_TRANS, KFE_NO_TRANS,
          M, N, K, alpha,
          static_cast<const float *>(A->descriptor.data), K,
          static_cast<const float *>(B->descriptor.data), N,
          static_cast<const float *>(bias_vec->descriptor.data),
          beta,
          static_cast<const float *>(residual->descriptor.data), N,
          activation,
          static_cast<float *>(C->descriptor.data), N,
          &kfe_stats);

      break;
    }

    default:
      return OG_ERR_UNSUPPORTED;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    group->execution_time_ms = elapsed.count();

    return ret;
  }

} // namespace og_internal

/* ========================================================================== */
/* Execute Single Node                                                        */
/* ========================================================================== */

int og_execute_node(og_graph_t graph, uint64_t node_id)
{
  if (!graph)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  NodeImpl *node = GetNode(impl, node_id);
  if (!node)
  {
    return OG_ERR_NODE_NOT_FOUND;
  }

  auto start = std::chrono::high_resolution_clock::now();

  int ret = OG_OK;
  switch (node->descriptor.type)
  {
  case OG_OP_GEMM:
    ret = ExecuteGemmNode(impl, node);
    break;
  case OG_OP_BIAS_ADD:
    ret = ExecuteBiasAddNode(impl, node);
    break;
  default:
    ret = OG_ERR_UNSUPPORTED;
    break;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  node->execution_time_ms = elapsed.count();
  node->is_executed = (ret == OG_OK);

  return ret;
}

int og_execute_fusion_group(og_graph_t graph,
                            const og_fusion_group_t *group,
                            og_graph_stats_t *stats)
{
  if (!graph || !group)
  {
    return OG_ERR_INVALID_ARG;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  // Find fusion group
  FusionGroupImpl *group_impl = nullptr;
  for (auto &gp : impl->fusion_groups)
  {
    if (gp.second->descriptor.group_id == group->group_id)
    {
      group_impl = gp.second.get();
      break;
    }
  }

  if (!group_impl)
  {
    return OG_ERR_NODE_NOT_FOUND;
  }

  int ret = ExecuteFusionGroupViaKFE(impl, group_impl);

  if (ret == OG_OK && stats)
  {
    stats->total_execution_time_ms = group_impl->execution_time_ms;
    stats->fusion_groups_detected = 1;
    stats->total_ops_fused = group->num_nodes;
    stats->total_memory_saved_bytes = group->estimated_memory_saved_bytes;
  }

  return ret;
}