// advanced/include/agee_internal.h
#ifndef JCORE_AGEE_INTERNAL_H_
#define JCORE_AGEE_INTERNAL_H_

#include "ag_execution_engine.h"

// Advanced component headers
#include "kernel_fusion_engine.h"
#include "operator_graph.h"

// Derived component headers
#include "cached_autotuner.h"
#include "global_thread_scheduler.h"
#include "microkernel_interface.h"
#include "pool_manager.h"

// Base component headers
#include "cpu_info.h"
#include "ffm_cache_block.h"

// Standard library headers
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace agee_internal {

  /* ========================================================================== */
/* Constants */
/* ========================================================================== */

constexpr size_t DEFAULT_POOL_SIZE_MB = 256;
constexpr size_t DEFAULT_WORKSPACE_SIZE_MB = 256;
constexpr double DEFAULT_FUSION_THRESHOLD = 1.2;
constexpr size_t MAX_FUSION_DEPTH = 8;
constexpr size_t PREFETCH_DISTANCE = 4;

/* ========================================================================== */
/* Forward Declarations */
/* ========================================================================== */

struct AGEEGlobalState;
struct SessionImpl;
struct GraphPlanImpl;

/* ========================================================================== */
/* Execution Node - Represents a schedulable OG-backed unit                   */
/* ========================================================================== */

struct ExecutionNode {
  /* ====================== Graph Identity ================================= */

  uint64_t node_id;          // Original AGEE graph node ID
  uint64_t og_node_id;       // Corresponding Operator Graph node ID

  og_op_type_t op_type;      // Operator Graph operation type

  /* ====================== Tensor Binding ================================= */

  std::vector<uint64_t> input_tensors;   // Tensor IDs (AGEE space)
  std::vector<uint64_t> output_tensors;  // Tensor IDs (AGEE space)

  /* ====================== Dependency Tracking ============================= */

  std::vector<uint64_t> dependencies;    // AGEE node dependencies

  /* ====================== Scheduling Metadata ============================= */

  int numa_node;                 // Assigned NUMA node (-1 = any)
  size_t thread_affinity;        // Preferred thread/core
  size_t estimated_flops;        // Estimated FLOPs (planner-level)
  size_t estimated_memory_bytes; // Estimated memory usage
  double estimated_time_ms;      // Estimated execution time

  /* ====================== OG / Fusion Metadata ============================ */

  uint64_t fusion_group_id;      // OG fusion group ID (0 = none)
  bool is_fused;                 // True if executed via OG fusion group

  /* ====================== Kernel Selection ================================ */

  std::string selected_kernel; // Pre-tuned kernel name
  void *kernel_params;         // Kernel-specific parameters

  /* ====================== Runtime State =================================== */

  bool is_ready;                 // Dependencies satisfied
  bool is_executed;              // Execution completed
  double actual_time_ms;         // Measured execution time

  ExecutionNode()
      : node_id(0),
        og_node_id(0),
        op_type(OG_OP_GEMM),
        numa_node(-1),
        thread_affinity(0),
        estimated_flops(0),
        estimated_memory_bytes(0),
        estimated_time_ms(0.0),
        fusion_group_id(0),
        kernel_params(nullptr),
        is_fused(false),
        is_ready(false),
        is_executed(false),
        actual_time_ms(0.0) {}
};

  /* ========================================================================== */
  /* Fusion Group - Operator Graph–owned fusion descriptor                      */
  /* ========================================================================== */

  struct FusionGroup {
    /* ====================== Identity ====================================== */

    uint64_t group_id;                 // AGEE fusion group ID

    og_graph_t og_graph;               // Owning Operator Graph
    og_fusion_group_t og_fusion_group; // OG fusion group handle

    og_fusion_pattern_t pattern;       // OG-detected fusion pattern (informational)

    /* ====================== Membership ==================================== */

    std::vector<uint64_t> node_ids;    // AGEE node IDs
    std::vector<uint64_t> input_tensors;
    std::vector<uint64_t> output_tensors;

    /* ====================== Pretuned Configuration ========================= */

    kfe_activation_t activation; // Activation function (if applicable)
    std::string selected_kernel;

    /* ====================== Scheduling Metadata ============================ */

    int numa_node;
    size_t estimated_memory_saved;
    double estimated_speedup;
    size_t estimated_flops;        // Total FLOPs for entire fusion group

    /* ====================== Runtime State ================================= */

    bool is_executed;
    double actual_time_ms;

    FusionGroup()
        : group_id(0),
          og_graph(nullptr),
          og_fusion_group(),
          pattern(OG_PATTERN_NONE),
          numa_node(-1),
          activation(KFE_ACTIVATION_NONE),
          estimated_flops(0),
          estimated_memory_saved(0),
          estimated_speedup(1.0),
          is_executed(false),
          actual_time_ms(0.0) {}
  };


/* ========================================================================== */
/* Tensor Descriptor - Memory and metadata for tensors                        */
/* ========================================================================== */

struct TensorDescriptor {
  uint64_t tensor_id;
  size_t ndim;
  size_t shape[8];
  size_t strides[8];
  void *data;
  size_t size_bytes;

  // Memory management - track allocation source
  enum class AllocSource {
    NONE,    // Not allocated by us
    POOL,    // Allocated from memory pool
    NUMA,    // Allocated via NUMA manager
    STANDARD // Allocated via ffm_malloc
  };

  AllocSource alloc_source; // Where memory came from
  int numa_node;            // Allocated NUMA node
  bool is_constant;         // Constant tensor (weights)
  bool is_owned;            // Whether we own this memory

  // Prefetch hints
  bool prefetch_enabled;
  size_t prefetch_distance;

  TensorDescriptor()
      : tensor_id(0), ndim(0), data(nullptr), size_bytes(0),
        alloc_source(AllocSource::NONE), numa_node(-1), is_constant(false),
        is_owned(false), prefetch_enabled(false), prefetch_distance(0) {
    memset(shape, 0, sizeof(shape));
    memset(strides, 0, sizeof(strides));
  }
};

/* ========================================================================== */
/* Graph Plan Implementation - Optimized execution plan                       */
/* ========================================================================== */

struct GraphPlanImpl {
  // Original graph handle
  og_graph_t original_graph;
  SessionImpl *owning_session;  // store the session when plan is created

  // Execution plan
  std::vector<ExecutionNode> execution_nodes;
  std::vector<FusionGroup> fusion_groups;
  std::vector<uint64_t> execution_order; // Topologically sorted

  // Tensor management
  std::unordered_map<uint64_t, TensorDescriptor> tensors;

  // Memory planning
  size_t peak_memory_bytes;
  size_t pooled_memory_bytes;
  std::vector<int> numa_node_assignments; // Per-tensor NUMA assignments

  // Statistics
  size_t total_operations;
  size_t fused_operations;
  double estimated_total_time_ms;

  // State
  bool is_optimized;
  bool memory_preallocated;

  // Mutex for thread-safe access
  mutable std::mutex plan_mutex;

  GraphPlanImpl()
      : original_graph(nullptr), peak_memory_bytes(0), pooled_memory_bytes(0),
        total_operations(0), fused_operations(0), estimated_total_time_ms(0.0),
        is_optimized(false), memory_preallocated(false) {}
};

/* ========================================================================== */
/* Session Implementation - Execution session with state                      */
/* ========================================================================== */

struct SessionImpl {
  uint64_t session_id;

  /* ====================== Core Runtime Components ======================= */

  pm_t *memory_pool;
  cat_handle_t *cached_tuner;
  jcore::global_thread::GlobalThreadScheduler *thread_scheduler;

  /* ====================== Operator Graph Runtime ======================== */

  og_graph_t operator_graph;     // <-- CENTRAL EXECUTION AUTHORITY

  /* ====================== Session Configuration ========================= */

  agee_config_t config;

  /* ====================== Hardware Information ========================== */

  cpu_info_t cpu_info;
  int max_numa_nodes;

  /* ====================== Execution State =============================== */

  std::vector<GraphPlanImpl *> active_plans;

  /* ====================== Statistics ==================================== */

  agee_exec_stats_t cumulative_stats;
  std::atomic<size_t> total_executions;

  /* ====================== Synchronization =============================== */

  mutable std::mutex session_mutex;

  /* ====================== Lifecycle ===================================== */

  SessionImpl()
      : session_id(0),
        memory_pool(nullptr),
        cached_tuner(nullptr),
        thread_scheduler(nullptr),
        operator_graph(nullptr),
        max_numa_nodes(0),
        total_executions(0) {

    memset(&config, 0, sizeof(config));
    memset(&cpu_info, 0, sizeof(cpu_info));
    memset(&cumulative_stats, 0, sizeof(cumulative_stats));
  }

  ~SessionImpl() {
    // operator_graph is destroyed explicitly in destroy_session
  }

  /* ====================== Plan Management =============================== */

  void UnregisterPlan(GraphPlanImpl *plan) {
    std::lock_guard<std::mutex> lock(session_mutex);
    auto it = std::find(active_plans.begin(), active_plans.end(), plan);
    if (it != active_plans.end())
      active_plans.erase(it);
  }
};


/* ========================================================================== */
/* Global State */
/* ========================================================================== */

struct AGEEGlobalState {
  bool initialized;
  agee_config_t global_config;

  // Component initialization flags
  bool kfe_initialized;
  bool og_initialized;
  bool vmath_initialized;
  bool profiler_initialized;
  bool dispatch_initialized;

  // Hardware information
  cpu_info_t cpu_info;
  ffm_cache_info_t *cache_info;

  // Session management
  std::vector<SessionImpl *> sessions;
  std::atomic<uint64_t> next_session_id;

  // Statistics
  std::atomic<size_t> total_graphs_executed;
  std::atomic<size_t> total_fusions_performed;

  // Mutex
  mutable std::mutex global_mutex;

  AGEEGlobalState()
      : initialized(false), kfe_initialized(false), og_initialized(false),
        vmath_initialized(false), profiler_initialized(false),
        dispatch_initialized(false), cache_info(nullptr), next_session_id(1),
        total_graphs_executed(0), total_fusions_performed(0) {
    memset(&global_config, 0, sizeof(global_config));
    memset(&cpu_info, 0, sizeof(cpu_info));
  }
};

// Global state instance (defined in agee_core.cpp)
extern AGEEGlobalState g_agee_state;

/* ========================================================================== */
/* Internal Function Declarations                                             */
/* ========================================================================== */

// agee_core.cpp
int InitializeComponents(const agee_config_t *config);
void ShutdownComponents();

// agee_graph_analysis.cpp
int AnalyzeGraph(SessionImpl *session, og_graph_t graph, GraphPlanImpl *plan);
int DetectFusionOpportunities(GraphPlanImpl *plan);
int BuildExecutionOrder(GraphPlanImpl *plan);

// agee_scheduler.cpp
int ScheduleOperations(SessionImpl *session, GraphPlanImpl *plan);
int AssignNUMANodes(SessionImpl *session, GraphPlanImpl *plan);
int AssignThreadAffinity(SessionImpl *session, GraphPlanImpl *plan);

// agee_executor.cpp
int ExecutePlanInternal(SessionImpl *session, GraphPlanImpl *plan,
                        agee_exec_stats_t *stats);
int ExecuteNode(SessionImpl *session, GraphPlanImpl *plan, ExecutionNode *node);
int ExecuteFusionGroup(SessionImpl *session, GraphPlanImpl *plan,
                       FusionGroup *group);

// agee_memory_manager.cpp
int PredictMemoryUsage(GraphPlanImpl *plan);
int AllocatePooledMemory(SessionImpl *session, GraphPlanImpl *plan);
int FreePlanMemory(SessionImpl *session, GraphPlanImpl *plan);
void *AllocateTensor(SessionImpl *session, size_t size_bytes, int numa_node);

/* ========================================================================== */
/* Utility Functions */
/* ========================================================================== */

inline double ComputeGFLOPS(double flops, double time_ms) {
  if (time_ms <= 0.0)
    return 0.0;
  return (flops / 1e9) / (time_ms / 1000.0);
}

inline size_t ComputeTensorSize(const og_tensor_t *tensor) {
  size_t total = 1;
  for (size_t i = 0; i < tensor->ndim; ++i) {
    total *= tensor->shape[i];
  }
  return total * sizeof(float); // Assuming float32
}

inline bool IsGEMMOperation(og_op_type_t type) {
  return type == OG_OP_GEMM || type == OG_OP_CONV2D;
}

inline bool IsActivationOperation(og_op_type_t type) {
  switch(type) {
    case OG_OP_RELU:
    case OG_OP_RELU6:
    case OG_OP_TANH:
    case OG_OP_SIGMOID:
    case OG_OP_GELU:
    case OG_OP_SWISH:
    case OG_OP_LEAKY_RELU:
      return true;
    default:
      return false;
  }
}

inline bool IsElementwiseOperation(og_op_type_t op) {
  switch(op) {
    case OG_OP_BIAS_ADD:
    case OG_OP_ELEMENTWISE_ADD:
    case OG_OP_ELEMENTWISE_MUL:
      return true;
    default:
      return false;
  }
}

inline kfe_activation_t ConvertActivationType(og_op_type_t type) {
  switch (type) {
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

} // namespace agee_internal

#endif /* JCORE_AGEE_INTERNAL_H_ */