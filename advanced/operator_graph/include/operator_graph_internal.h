// advanced/operator_graph/include/operator_graph_internal.h
#ifndef JCORE_OPERATOR_GRAPH_INTERNAL_H_
#define JCORE_OPERATOR_GRAPH_INTERNAL_H_

#include "operator_graph.h"
#include "kernel_fusion_engine.h"
#include "k_kernel_dispatch.h"
#include "global_thread_scheduler.h"
#include "cpu_info.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <chrono>

namespace og_internal
{

  /* ========================================================================== */
  /* Internal Data Structures                                                    */
  /* ========================================================================== */

  extern std::unordered_map<void*, bool> tensor_allocation_map; // true = pool, false = posix_memalign
  extern std::mutex tensor_alloc_mutex;

  /**
   * @brief Internal tensor representation with reference counting
   */
  struct TensorImpl
  {
    og_tensor_t descriptor;
    std::atomic<int> ref_count{0};
    bool is_alive{true};
    void *allocated_data{nullptr}; // If we allocated it internally
    size_t allocated_size{0};

    TensorImpl() = default;
    explicit TensorImpl(const og_tensor_t &desc) : descriptor(desc) {}
  };

  /**
   * @brief Internal node representation with adjacency lists
   */
  struct NodeImpl
  {
    og_node_t descriptor;
    std::vector<uint64_t> input_tensor_ids;
    std::vector<uint64_t> output_tensor_ids;
    std::vector<uint64_t> successor_nodes;   // Nodes that depend on this
    std::vector<uint64_t> predecessor_nodes; // Nodes this depends on
    int in_degree{0};                        // For topological sort
    bool is_executed{false};
    double execution_time_ms{0.0};

    NodeImpl() = default;
    explicit NodeImpl(const og_node_t &desc) : descriptor(desc) {}
  };

  /**
   * @brief Internal fusion group representation
   */
  struct FusionGroupImpl
  {
    og_fusion_group_t descriptor;
    std::vector<uint64_t> node_sequence; // Execution order within group
    std::vector<uint64_t> input_tensors;
    std::vector<uint64_t> output_tensors;
    bool is_executed{false};
    double execution_time_ms{0.0};

    FusionGroupImpl() = default;
    explicit FusionGroupImpl(const og_fusion_group_t &desc) : descriptor(desc) {}
  };

  /* ========================================================================== */
  /* Graph Implementation                                                        */
  /* ========================================================================== */

  struct GraphImpl
  {
    // Core graph data
    std::unordered_map<uint64_t, std::unique_ptr<TensorImpl>> tensors;
    std::unordered_map<uint64_t, std::unique_ptr<NodeImpl>> nodes;
    std::unordered_map<uint64_t, std::unique_ptr<FusionGroupImpl>> fusion_groups;

    // Graph state
    bool is_finalized{false};
    bool is_optimized{false};
    std::vector<uint64_t> topological_order;
    std::vector<uint64_t> fusion_execution_order;

    // Statistics
    og_graph_stats_t stats{};
    std::atomic<uint64_t> next_tensor_id{1};
    std::atomic<uint64_t> next_node_id{1};
    std::atomic<uint64_t> next_fusion_group_id{1};

    // Thread safety
    mutable std::mutex graph_mutex;

    // Performance tracking
    std::chrono::high_resolution_clock::time_point last_execution_start;
    std::chrono::high_resolution_clock::time_point last_execution_end;

    GraphImpl() = default;
    ~GraphImpl() = default;

    // Delete copy constructor and assignment
    GraphImpl(const GraphImpl &) = delete;
    GraphImpl &operator=(const GraphImpl &) = delete;
  };

  /* ========================================================================== */
  /* Global Runtime State                                                        */
  /* ========================================================================== */

  struct OGRuntimeState
  {
    bool initialized{false};
    og_config_t config{};
    std::mutex state_mutex;

    // Component handles
    jcore::global_thread::GlobalThreadScheduler *scheduler{nullptr};
    cpu_info_t cpu_info{};
    ffm_cache_info_t *cache_info{};
    pm_t *pool_manager{};

    // Statistics
    std::atomic<size_t> total_graphs_created{0};
    std::atomic<size_t> total_graphs_executed{0};
    std::atomic<size_t> total_fusion_groups_executed{0};

    OGRuntimeState() = default;
    ~OGRuntimeState() = default;
  };

  // Global state instance (defined in operator_graph_core.cpp)
  extern OGRuntimeState g_og_state;

  /* ========================================================================== */
  /* Graph Analysis Functions                                                    */
  /* ========================================================================== */

  /**
   * @brief Perform topological sort on graph
   * @return OG_OK on success, OG_ERR_CYCLE_DETECTED if cycle found
   */
  int TopologicalSort(GraphImpl *graph);

  /**
   * @brief Detect cycles in graph using DFS
   * @return true if cycle exists
   */
  bool HasCycle(GraphImpl *graph);

  /**
   * @brief Validate all tensor references in nodes
   * @return OG_OK on success
   */
  int ValidateTensorReferences(GraphImpl *graph);

  /**
  * @brief Build adjacency lists by analyzing tensor data flow
  *
  * Automatically constructs edges between nodes by analyzing which nodes
  * produce outputs that are consumed as inputs by other nodes.
  *
  * NOTE: This function clears and rebuilds ALL edges. For graphs with manual edges, call og_add_edge() AFTER
  * og_finalize_graph() or skip BuildAdjacencyLists() entirely.
  *
  * @param graph Graph implementation
  */
  void BuildAdjacencyLists(GraphImpl *graph);

  /* ========================================================================== */
  /* Pattern Detection Functions                                                 */
  /* ========================================================================== */

  /**
   * @brief Detect GEMM + Bias pattern
   */
  bool DetectGemmBiasPattern(GraphImpl *graph, uint64_t node_id,
                             og_fusion_group_t *out_group);

  /**
   * @brief Detect GEMM + Bias + Activation pattern
   */
  bool DetectGemmBiasActivationPattern(GraphImpl *graph, uint64_t node_id,
                                       og_fusion_group_t *out_group);

  /**
   * @brief Detect GEMM + Residual + Activation pattern
   */
  bool DetectGemmResidualPattern(GraphImpl *graph, uint64_t node_id,
                                 og_fusion_group_t *out_group);

  /**
   * @brief Detect linear GEMM chains
   */
  bool DetectLinearChainPattern(GraphImpl *graph, uint64_t node_id,
                                og_fusion_group_t *out_group);

  /**
   * @brief Main pattern detection orchestrator
   */
  int AnalyzeAndDetectPatterns(GraphImpl *graph);

  /* ========================================================================== */
  /* Execution Functions                                                         */
  /* ========================================================================== */

  /**
   * @brief Execute a single GEMM node
   */
  int ExecuteGemmNode(GraphImpl *graph, NodeImpl *node);

  /**
   * @brief Execute a bias addition node
   */
  int ExecuteBiasAddNode(GraphImpl *graph, NodeImpl *node);

  /**
   * @brief Execute an activation node
   */
  int ExecuteActivationNode(GraphImpl *graph, NodeImpl *node);

  /**
   * @brief Execute element-wise operation node
   */
  int ExecuteElementwiseNode(GraphImpl *graph, NodeImpl *node);

  /**
   * @brief Execute a fusion group via KFE
   */
  int ExecuteFusionGroupViaKFE(GraphImpl *graph, FusionGroupImpl *group);

  /* ========================================================================== */
  /* Memory Management Functions                                                 */
  /* ========================================================================== */

  /**
   * @brief Allocate tensor data
   */
  void *AllocateTensorData(size_t size_bytes);

  /**
   * @brief Free tensor data
   */
  void FreeTensorData(void *ptr, size_t size_bytes);

  /**
   * @brief Estimate peak memory usage for graph
   */
  size_t EstimatePeakMemory(GraphImpl *graph);

  /**
   * @brief Plan memory reuse opportunities
   */
  int PlanMemoryReuse(GraphImpl *graph);

  /* ========================================================================== */
  /* Utility Functions                                                           */
  /* ========================================================================== */

  /**
   * @brief Check if operation is an activation function
   */
  inline bool IsActivationOp(og_op_type_t type)
  {
    return type >= OG_OP_RELU && type <= OG_OP_LEAKY_RELU;
  }

  /**
   * @brief Check if operation is element-wise
   */
  inline bool IsElementwiseOp(og_op_type_t type)
  {
    return type == OG_OP_ELEMENTWISE_ADD || type == OG_OP_ELEMENTWISE_MUL ||
           type == OG_OP_BIAS_ADD || IsActivationOp(type);
  }

  /**
   * @brief Convert og_op_type_t to kfe_activation_t
   */
  inline kfe_activation_t OpTypeToKFEActivation(og_op_type_t type)
  {
    switch (type)
    {
    case OG_OP_RELU:
      return KFE_ACTIVATION_RELU;
    case OG_OP_RELU6:
      return KFE_ACTIVATION_RELU6;
    case OG_OP_TANH:
      return KFE_ACTIVATION_TANH;
    case OG_OP_SIGMOID:
      return KFE_ACTIVATION_SIGMOID;
    case OG_OP_GELU:
      return KFE_ACTIVATION_GELU;
    case OG_OP_SWISH:
      return KFE_ACTIVATION_SWISH;
    case OG_OP_LEAKY_RELU:
      return KFE_ACTIVATION_LEAKY_RELU;
    default:
      return KFE_ACTIVATION_NONE;
    }
  }

  /**
   * @brief Calculate theoretical FLOPS for operation
   */
  inline double CalculateFLOPS(og_op_type_t type, const og_tensor_t *inputs,
                               size_t num_inputs, const og_tensor_t *outputs,
                               size_t num_outputs)
  {
    if (type == OG_OP_GEMM && num_inputs >= 2)
    {
      // GEMM: 2*M*N*K FLOPs
      size_t M = inputs[0].shape[0];
      size_t K = inputs[0].shape[1];
      size_t N = inputs[1].shape[1];
      return 2.0 * M * N * K;
    }
    else if (IsElementwiseOp(type) && num_outputs > 0)
    {
      // Element-wise: 1 FLOP per element
      return static_cast<double>(outputs[0].total_elements);
    }
    return 0.0;
  }

  /**
   * @brief Compute GFLOPS from FLOPS and time
   */
  inline double ComputeGFLOPS(double flops, double time_ms)
  {
    if (time_ms <= 0.0)
      return 0.0;
    return (flops / 1e9) / (time_ms / 1000.0);
  }

  /**
   * @brief Get tensor by ID
   */
  inline TensorImpl *GetTensor(GraphImpl *graph, uint64_t tensor_id)
  {
    auto it = graph->tensors.find(tensor_id);
    return (it != graph->tensors.end()) ? it->second.get() : nullptr;
  }

  /**
   * @brief Get node by ID
   */
  inline NodeImpl *GetNode(GraphImpl *graph, uint64_t node_id)
  {
    auto it = graph->nodes.find(node_id);
    return (it != graph->nodes.end()) ? it->second.get() : nullptr;
  }

} // namespace og_internal

#endif /* JCORE_OPERATOR_GRAPH_INTERNAL_H_ */