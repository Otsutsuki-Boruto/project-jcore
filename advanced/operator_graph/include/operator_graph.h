// advanced/operator_graph/include/operator_graph.h
#ifndef JCORE_OPERATOR_GRAPH_H_
#define JCORE_OPERATOR_GRAPH_H_

/**
 * @file operator_graph.h
 * @brief Operator Graph / Fusion Runtime - Detects fusible operation patterns
 *
 * Component: Operator Graph / Fusion Runtime (Advanced)
 * Purpose: Analyze computational graphs and detect fusion opportunities
 *
 * Dependencies:
 *   Advanced:
 *     - Kernel Fusion Engine: Execute fused operations
 *   Derived:
 *     - Kernel Dispatch Table / Runtime Selector: Fallback kernel execution
 *     - Global Thread & Task Scheduler: Parallel graph traversal
 *   Base:
 *     - Microbenchmark & Timer Utilities: Performance measurement
 *     - CPU Feature Detection Module: Hardware capabilities
 *
 * Capabilities:
 *   - Build directed acyclic graphs (DAGs) of operations
 *   - Detect fusible operation patterns (GEMM chains, activation sequences)
 *   - Analyze data dependencies and memory reuse opportunities
 *   - Coordinate with Kernel Fusion Engine for execution
 *   - Provide fusion statistics and telemetry
 *
 * Thread-safety: Thread-safe after initialization
 * FFM API: Fully compatible with Project JCore FFM API
 */

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Error Codes */
/* ========================================================================== */

#define OG_OK 0
#define OG_ERR_NOT_INITIALIZED -1
#define OG_ERR_INVALID_ARG -2
#define OG_ERR_NO_MEMORY -3
#define OG_ERR_INTERNAL -4
#define OG_ERR_CYCLE_DETECTED -5
#define OG_ERR_NODE_NOT_FOUND -6
#define OG_ERR_INVALID_GRAPH -7
#define OG_ERR_FUSION_FAILED -8
#define OG_ERR_UNSUPPORTED -9

/* ========================================================================== */
/* Operation Types */
/* ========================================================================== */

typedef enum {
  OG_OP_GEMM = 0,            /**< Matrix multiplication */
  OG_OP_BIAS_ADD = 1,        /**< Bias addition (broadcast) */
  OG_OP_ELEMENTWISE_ADD = 2, /**< Element-wise addition */
  OG_OP_ELEMENTWISE_MUL = 3, /**< Element-wise multiplication */
  OG_OP_RELU = 4,            /**< ReLU activation */
  OG_OP_RELU6 = 5,           /**< ReLU6 activation */
  OG_OP_TANH = 6,            /**< Tanh activation */
  OG_OP_SIGMOID = 7,         /**< Sigmoid activation */
  OG_OP_GELU = 8,            /**< GELU activation */
  OG_OP_SWISH = 9,           /**< Swish activation */
  OG_OP_LEAKY_RELU = 10,     /**< Leaky ReLU activation */
  OG_OP_BATCH_NORM = 11,     /**< Batch normalization */
  OG_OP_LAYER_NORM = 12,     /**< Layer normalization */
  OG_OP_SOFTMAX = 13,        /**< Softmax activation */
  OG_OP_DROPOUT = 14,        /**< Dropout (training only) */
  OG_OP_CONV2D = 15,         /**< 2D Convolution */
  OG_OP_POOLING = 16,        /**< Pooling operation */
  OG_OP_CONCAT = 17,         /**< Concatenation */
  OG_OP_SPLIT = 18,          /**< Split operation */
  OG_OP_RESHAPE = 19,        /**< Reshape (view change) */
  OG_OP_TRANSPOSE = 20,      /**< Matrix transpose */
  OG_OP_CUSTOM = 99          /**< Custom user-defined operation */
} og_op_type_t;

/* ========================================================================== */
/* Fusion Pattern Types */
/* ========================================================================== */

typedef enum {
  OG_PATTERN_NONE = 0,                 /**< No fusion pattern detected */
  OG_PATTERN_GEMM_BIAS = 1,            /**< GEMM + Bias */
  OG_PATTERN_GEMM_BIAS_ACTIVATION = 2, /**< GEMM + Bias + Activation */
  OG_PATTERN_GEMM_ADD = 3,             /**< GEMM + Element-wise Add */
  OG_PATTERN_GEMM_RESIDUAL_ACTIVATION =
      4, /**< GEMM + Bias + Residual + Activation */
  OG_PATTERN_CONV_BIAS_ACTIVATION = 5, /**< Conv2D + Bias + Activation */
  OG_PATTERN_LINEAR_CHAIN = 6,         /**< Multiple GEMMs in sequence */
  OG_PATTERN_BATCH_GEMM = 7,           /**< Batched GEMM operations */
  OG_PATTERN_ACTIVATION_CHAIN = 8      /**< Multiple activations in sequence */
} og_fusion_pattern_t;

/* ========================================================================== */
/* Tensor Descriptor */
/* ========================================================================== */

#define OG_MAX_TENSOR_DIMS 8

typedef struct {
  uint64_t id;                        /**< Unique tensor identifier */
  size_t ndim;                        /**< Number of dimensions */
  size_t shape[OG_MAX_TENSOR_DIMS];   /**< Dimension sizes */
  size_t strides[OG_MAX_TENSOR_DIMS]; /**< Memory strides */
  void *data;            /**< Pointer to tensor data (can be NULL) */
  size_t total_elements; /**< Total number of elements */
  size_t size_bytes;     /**< Total size in bytes */
  int is_constant;       /**< 1 if constant (weights), 0 if variable */
} og_tensor_t;

/* ========================================================================== */
/* Operation Node Descriptor */
/* ========================================================================== */

#define OG_MAX_INPUTS 8
#define OG_MAX_OUTPUTS 4
#define OG_MAX_ATTRIBUTES 16

typedef struct {
  uint64_t id;       /**< Unique node identifier */
  og_op_type_t type; /**< Operation type */
  const char *name;  /**< Human-readable name (optional) */

  uint64_t input_ids[OG_MAX_INPUTS]; /**< Input tensor IDs */
  size_t num_inputs;                 /**< Number of inputs */

  uint64_t output_ids[OG_MAX_OUTPUTS]; /**< Output tensor IDs */
  size_t num_outputs;                  /**< Number of outputs */

  // Operation-specific attributes (e.g., alpha, beta, transpose flags)
  float attributes[OG_MAX_ATTRIBUTES]; /**< Numeric attributes */
  size_t num_attributes;               /**< Number of attributes */

  int can_fuse_forward;  /**< Can fuse with next operation */
  int can_fuse_backward; /**< Can fuse with previous operation */
  int is_executed;       /**< Execution status flag */
} og_node_t;

/* ========================================================================== */
/* Fusion Group Descriptor */
/* ========================================================================== */

#define OG_MAX_FUSION_NODES 16

typedef struct {
  uint64_t group_id;                      /**< Unique fusion group ID */
  og_fusion_pattern_t pattern;            /**< Detected fusion pattern */
  uint64_t node_ids[OG_MAX_FUSION_NODES]; /**< Node IDs in this group */
  size_t num_nodes;                       /**< Number of nodes in group */
  size_t
      estimated_memory_saved_bytes; /**< Estimated memory traffic reduction */
  double estimated_speedup;         /**< Estimated performance speedup */
  int is_executed;                  /**< Execution status */
} og_fusion_group_t;

/* ========================================================================== */
/* Graph Statistics */
/* ========================================================================== */

typedef struct {
  size_t total_nodes;              /**< Total number of operation nodes */
  size_t total_tensors;            /**< Total number of tensors */
  size_t total_edges;              /**< Total number of edges (dependencies) */
  size_t fusion_groups_detected;   /**< Number of fusion groups found */
  size_t total_ops_fused;          /**< Total operations successfully fused */
  size_t total_memory_saved_bytes; /**< Total memory traffic eliminated */
  double total_execution_time_ms;  /**< Total graph execution time */
  double avg_fusion_speedup;       /**< Average speedup from fusion */
  const char *bottleneck_op;       /**< Most time-consuming operation */
} og_graph_stats_t;

/* ========================================================================== */
/* Configuration Structure */
/* ========================================================================== */

typedef struct {
  int enable_fusion;           /**< Enable fusion optimization (1/0) */
  int enable_parallelism;      /**< Enable parallel graph execution (1/0) */
  int enable_memory_reuse;     /**< Enable tensor memory reuse (1/0) */
  int enable_pattern_matching; /**< Enable pattern detection (1/0) */
  int max_fusion_depth;        /**< Maximum depth for fusion chains */
  size_t num_threads;          /**< Number of threads (0 = auto) */
  int verbose;                 /**< Verbose logging (1/0) */
  double fusion_threshold;     /**< Minimum speedup threshold for fusion */
} og_config_t;

/* ========================================================================== */
/* Opaque Graph Handle */
/* ========================================================================== */

typedef struct og_graph_impl *og_graph_t;

/* ========================================================================== */
/* Initialization & Configuration */
/* ========================================================================== */

/**
 * @brief Initialize Operator Graph Runtime
 *
 * Initializes all dependencies (KFE, dispatch table, scheduler, etc.)
 *
 * @param config Configuration structure (NULL = use defaults)
 * @return OG_OK on success, error code otherwise
 */
int og_init(const og_config_t *config);

/**
 * @brief Shutdown Operator Graph Runtime
 *
 * Cleanup resources and finalize all subsystems.
 */
void og_shutdown(void);

/**
 * @brief Check if Operator Graph Runtime is initialized
 *
 * @return 1 if initialized, 0 otherwise
 */
int og_is_initialized(void);

/* ========================================================================== */
/* Graph Construction */
/* ========================================================================== */

/**
 * @brief Create a new empty computational graph
 *
 * @param out_graph Output graph handle
 * @return OG_OK on success
 */
int og_create_graph(og_graph_t *out_graph);

/**
 * @brief Destroy a graph and free all associated resources
 *
 * @param graph Graph handle to destroy
 */
void og_destroy_graph(og_graph_t graph);

/**
 * @brief Add a tensor to the graph
 *
 * @param graph Graph handle
 * @param tensor Tensor descriptor
 * @param out_tensor_id Output tensor ID
 * @return OG_OK on success
 */
int og_add_tensor(og_graph_t graph, const og_tensor_t *tensor,
                  uint64_t *out_tensor_id);

/**
 * @brief Add an operation node to the graph
 *
 * @param graph Graph handle
 * @param node Node descriptor
 * @param out_node_id Output node ID
 * @return OG_OK on success
 */
int og_add_node(og_graph_t graph, const og_node_t *node, uint64_t *out_node_id);

/**
 * @brief Add a dependency edge between two nodes
 *
 * @param graph Graph handle
 * @param from_node Source node ID
 * @param to_node Destination node ID
 * @return OG_OK on success
 */
int og_add_edge(og_graph_t graph, uint64_t from_node, uint64_t to_node);

/**
 * @brief Finalize graph construction (validate and optimize)
 *
 * Performs:
 * - Cycle detection
 * - Topological sorting
 * - Fusion pattern detection
 * - Memory planning
 *
 * @param graph Graph handle
 * @return OG_OK on success
 */
int og_finalize_graph(og_graph_t graph);

/* ========================================================================== */
/* Pattern Detection & Fusion Analysis */
/* ========================================================================== */

/**
 * @brief Analyze graph and detect fusible patterns
 *
 * @param graph Graph handle
 * @param out_groups Output array of detected fusion groups
 * @param max_groups Maximum number of groups to return
 * @param out_num_groups Number of groups detected
 * @return OG_OK on success
 */
int og_detect_fusion_patterns(og_graph_t graph, og_fusion_group_t *out_groups,
                              size_t max_groups, size_t *out_num_groups);

/**
 * @brief Get fusion opportunities for a specific node
 *
 * @param graph Graph handle
 * @param node_id Node ID to analyze
 * @param out_pattern Detected fusion pattern
 * @param out_group Fusion group (if applicable)
 * @return OG_OK on success
 */
int og_get_fusion_info(og_graph_t graph, uint64_t node_id,
                       og_fusion_pattern_t *out_pattern,
                       og_fusion_group_t *out_group);

/* ========================================================================== */
/* Graph Execution */
/* ========================================================================== */

/**
 * @brief Execute the entire computational graph
 *
 * Performs topological execution with fusion optimization.
 *
 * @param graph Graph handle
 * @param stats Optional performance statistics (NULL = don't collect)
 * @return OG_OK on success
 */
int og_execute_graph(og_graph_t graph, og_graph_stats_t *stats);

/**
 * @brief Execute a specific fusion group
 *
 * @param graph Graph handle
 * @param group Fusion group to execute
 * @param stats Optional performance statistics
 * @return OG_OK on success
 */
int og_execute_fusion_group(og_graph_t graph, const og_fusion_group_t *group,
                            og_graph_stats_t *stats);

/**
 * @brief Execute a single node (no fusion)
 *
 * @param graph Graph handle
 * @param node_id Node ID to execute
 * @return OG_OK on success
 */
int og_execute_node(og_graph_t graph, uint64_t node_id);

/* ========================================================================== */
/* Graph Query & Inspection */
/* ========================================================================== */

/**
 * @brief Get graph statistics
 *
 * @param graph Graph handle
 * @param stats Output statistics structure
 * @return OG_OK on success
 */
int og_get_graph_stats(og_graph_t graph, og_graph_stats_t *stats);

/**
 * @brief Get node information
 *
 * @param graph Graph handle
 * @param node_id Node ID
 * @param out_node Output node descriptor
 * @return OG_OK on success
 */
int og_get_node_info(og_graph_t graph, uint64_t node_id, og_node_t *out_node);

/**
 * @brief Get tensor information
 *
 * @param graph Graph handle
 * @param tensor_id Tensor ID
 * @param out_tensor Output tensor descriptor
 * @return OG_OK on success
 */
int og_get_tensor_info(og_graph_t graph, uint64_t tensor_id,
                       og_tensor_t *out_tensor);

/**
 * @brief Get topological execution order
 *
 * @param graph Graph handle
 * @param out_order Output array of node IDs
 * @param max_nodes Maximum number of nodes
 * @param out_num_nodes Number of nodes in execution order
 * @return OG_OK on success
 */
int og_get_execution_order(og_graph_t graph, uint64_t *out_order,
                           size_t max_nodes, size_t *out_num_nodes);

/* ========================================================================== */
/* Graph Optimization */
/* ========================================================================== */

/**
 * @brief Optimize graph for execution
 *
 * Applies:
 * - Dead code elimination
 * - Constant folding
 * - Memory layout optimization
 * - Fusion opportunities
 *
 * @param graph Graph handle
 * @return OG_OK on success
 */
int og_optimize_graph(og_graph_t graph);

/**
 * @brief Estimate memory footprint of graph execution
 *
 * @param graph Graph handle
 * @param out_peak_memory_bytes Peak memory usage in bytes
 * @return OG_OK on success
 */
int og_estimate_memory_usage(og_graph_t graph, size_t *out_peak_memory_bytes);

/* ========================================================================== */
/* Utility Functions */
/* ========================================================================== */

/**
 * @brief Get operation type name
 *
 * @param op_type Operation type
 * @return Human-readable string
 */
const char *og_op_type_name(og_op_type_t op_type);

/**
 * @brief Get fusion pattern name
 *
 * @param pattern Fusion pattern type
 * @return Human-readable string
 */
const char *og_pattern_name(og_fusion_pattern_t pattern);

/**
 * @brief Convert error code to string
 *
 * @param error Error code
 * @return Error message string
 */
const char *og_strerror(int error);

/**
 * @brief Print graph structure (for debugging)
 *
 * @param graph Graph handle
 * @param verbose Detailed output (1/0)
 */
void og_print_graph(og_graph_t graph, int verbose);

/**
 * @brief Export graph to DOT format (Graphviz)
 *
 * @param graph Graph handle
 * @param filename Output file path
 * @return OG_OK on success
 */
int og_export_graph_dot(og_graph_t graph, const char *filename);

/**
 * @brief Validate graph integrity
 *
 * Checks for:
 * - Cycles
 * - Invalid tensor references
 * - Type mismatches
 *
 * @param graph Graph handle
 * @return OG_OK if valid, error code otherwise
 */
int og_validate_graph(og_graph_t graph);

/**
 * @brief Reset graph execution state for re-execution
 *
 * Clears execution flags on all nodes and fusion groups,
 * allowing the graph to be executed multiple times.
 *
 * @param graph Graph handle
 * @return OG_OK on success
 */
int og_reset_execution_state(og_graph_t graph);

#ifdef __cplusplus
}
#endif

#endif /* JCORE_OPERATOR_GRAPH_H_ */