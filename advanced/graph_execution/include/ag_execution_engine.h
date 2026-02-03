// advanced/include/ag_execution_engine.h
#ifndef JCORE_ADAPTIVE_GRAPH_EXECUTION_ENGINE_H_
#define JCORE_ADAPTIVE_GRAPH_EXECUTION_ENGINE_H_

/**
 * @file ag_execution_engine.h
 * @brief Adaptive Graph Execution Engine - Graph-level optimization and
 * execution
 *
 * Component: Adaptive Graph Execution Engine (Advanced)
 * Purpose: Maximizes hardware utilization through predictive scheduling and
 *          pre-tuned execution plans
 *
 * Dependencies:
 *   Advanced:
 *     - Kernel Fusion Engine: Execute fused operations
 *     - Operator Graph / Fusion Runtime: Graph analysis and pattern detection
 *   Derived:
 *     - Adaptive Kernel Auto-Tuner: Pre-tuned kernel selection
 *     - NUMA-Aware Memory Manager: NUMA-optimized memory allocation
 *     - Memory Pool Manager: Pooled memory allocation
 *     - Kernel Dispatch Table / Runtime Selector: Kernel execution
 *     - Performance Profiler / Telemetry: Performance tracking
 *     - Global Thread & Task Scheduler: Thread management
 *     - Vector Math Engine (SLEEF/xsimd): Vectorized operations
 *     - Microkernel Interface Layer: Low-level GEMM operations
 *     - Tuning Result Cache System: Cached tuning results
 *   Base:
 *     - CPU Feature Detection Module: Hardware capability detection
 *     - ISA-Aware Dispatch Mechanism: ISA-specific dispatch
 *     - Hardware Introspection Layer: Topology information
 *     - Memory Allocator Wrapper: Base memory allocation
 *     - Huge Page Controller: Large page support
 *     - Memory Prefetch Interface: Cache-aware prefetching
 *     - Cache Blocking / Tiling Utility: Optimal tile sizes
 *     - Base Thread Scheduler Abstraction: Thread control
 *     - Microbenchmark & Timer Utilities: Performance measurement
 *     - Configuration & Env Controller: Configuration management
 *
 * Capabilities:
 *   - Graph-wide fusion planning and optimization
 *   - Predictive execution scheduling with NUMA awareness
 *   - Pre-tuned kernel selection based on hardware and workload
 *   - Vectorized element-wise operations integration
 *   - Memory and resource prediction
 *   - Session-based adaptation and tuning
 *   - Batched graph execution with parallelism
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

#define AGEE_OK 0
#define AGEE_ERR_NOT_INITIALIZED -1
#define AGEE_ERR_INVALID_ARG -2
#define AGEE_ERR_NO_MEMORY -3
#define AGEE_ERR_INTERNAL -4
#define AGEE_ERR_GRAPH_INVALID -5
#define AGEE_ERR_EXECUTION_FAILED -6
#define AGEE_ERR_OOM -7
#define AGEE_ERR_UNSUPPORTED -8

/* ========================================================================== */
/* Configuration Structure */
/* ========================================================================== */

typedef struct {
  size_t num_threads;           /**< Number of execution threads (0 = auto) */
  int enable_fusion;            /**< Enable kernel fusion (1/0) */
  int enable_numa_optimization; /**< Enable NUMA-aware scheduling (1/0) */
  int enable_prefetch;          /**< Enable memory prefetching (1/0) */
  int enable_adaptive_tuning;   /**< Enable adaptive kernel selection (1/0) */
  int enable_memory_pooling;    /**< Enable memory pool manager (1/0) */
  int use_hugepages;            /**< Use huge pages for allocations (1/0) */
  size_t memory_pool_size_mb;   /**< Memory pool size in MB (0 = auto) */
  size_t workspace_size_mb;     /**< Workspace size in MB (0 = auto) */
  double fusion_threshold; /**< Minimum speedup for fusion (default: 1.2) */
  int verbose;             /**< Verbose logging (1/0) */
  int profile_execution;   /**< Enable detailed profiling (1/0) */
} agee_config_t;

/* ========================================================================== */
/* Execution Statistics */
/* ========================================================================== */

typedef struct {
  double total_execution_time_ms;   /**< Total graph execution time */
  double scheduling_overhead_ms;    /**< Time spent on scheduling */
  double fusion_time_ms;            /**< Time spent in fused operations */
  double kernel_selection_time_ms;  /**< Time spent selecting kernels */
  size_t total_operations;          /**< Total number of operations */
  size_t fused_operations;          /**< Number of fused operations */
  size_t memory_allocated_bytes;    /**< Total memory allocated */
  size_t memory_saved_bytes;        /**< Memory saved through fusion */
  double achieved_gflops;           /**< Achieved GFLOPS */
  double memory_bandwidth_gbps;     /**< Memory bandwidth utilized */
  double fusion_speedup;            /**< Speedup from fusion */
  const char *bottleneck_operation; /**< Slowest operation type */
} agee_exec_stats_t;

/* ========================================================================== */
/* Session Handle (Opaque) */
/* ========================================================================== */

typedef struct agee_session_impl *agee_session_t;

/* ========================================================================== */
/* Graph Plan Handle (Opaque) */
/* ========================================================================== */

typedef struct agee_graph_plan_impl *agee_graph_plan_t;

/* ========================================================================== */
/* Initialization & Configuration */
/* ========================================================================== */

/**
 * @brief Initialize Adaptive Graph Execution Engine
 *
 * Initializes all dependencies and prepares the execution environment.
 * Must be called before any other AGEE functions.
 *
 * @param config Configuration structure (NULL = use defaults)
 * @return AGEE_OK on success, error code otherwise
 */
int agee_init(const agee_config_t *config);

/**
 * @brief Shutdown Adaptive Graph Execution Engine
 *
 * Cleanup all resources and finalize subsystems.
 * Safe to call multiple times.
 */
void agee_shutdown(void);

/**
 * @brief Check if AGEE is initialized
 *
 * @return 1 if initialized, 0 otherwise
 */
int agee_is_initialized(void);

/**
 * @brief Get default configuration
 *
 * @param out_config Output configuration structure
 * @return AGEE_OK on success
 */
int agee_get_default_config(agee_config_t *out_config);

/* ========================================================================== */
/* Session Management */
/* ========================================================================== */

/**
 * @brief Create a new execution session
 *
 * Sessions maintain state for adaptive tuning and caching.
 * Multiple sessions can coexist for different workloads.
 *
 * @param out_session Output session handle
 * @return AGEE_OK on success
 */
int agee_create_session(agee_session_t *out_session);

/**
 * @brief Destroy an execution session
 *
 * @param session Session handle to destroy
 */
void agee_destroy_session(agee_session_t session);

/**
 * @brief Reset session state (clear cached plans)
 *
 * @param session Session handle
 * @return AGEE_OK on success
 */
int agee_reset_session(agee_session_t session);

/* ========================================================================== */
/* Graph Planning */
/* ========================================================================== */

/**
 * @brief Create an execution plan from an operator graph
 *
 * Analyzes the graph, detects fusion opportunities, and creates
 * an optimized execution plan with pre-tuned kernels.
 *
 * @param session Session handle
 * @param graph Operator graph handle (from operator_graph.h)
 * @param out_plan Output execution plan handle
 * @return AGEE_OK on success
 */
int agee_create_plan_from_graph(agee_session_t session, void *graph,
                                agee_graph_plan_t *out_plan);

/**
 * @brief Optimize an existing execution plan
 *
 * Re-analyzes and re-tunes the plan based on runtime statistics.
 *
 * @param session Session handle
 * @param plan Plan handle to optimize
 * @return AGEE_OK on success
 */
int agee_optimize_plan(agee_session_t session, agee_graph_plan_t plan);

/**
 * @brief Destroy an execution plan
 *
 * @param plan Plan handle to destroy
 */
void agee_destroy_plan(agee_graph_plan_t plan);

/* ========================================================================== */
/* Execution */
/* ========================================================================== */

/**
 * @brief Execute a graph plan
 *
 * Executes the optimized plan using predictive scheduling and
 * pre-tuned kernels.
 *
 * @param session Session handle
 * @param plan Execution plan
 * @param stats Optional execution statistics (NULL = don't collect)
 * @return AGEE_OK on success
 */
int agee_execute_plan(agee_session_t session, agee_graph_plan_t plan,
                      agee_exec_stats_t *stats);

/**
 * @brief Execute a graph plan with input tensors
 *
 * @param session Session handle
 * @param plan Execution plan
 * @param input_tensors Array of input tensor pointers
 * @param num_inputs Number of input tensors
 * @param output_tensors Array of output tensor pointers (pre-allocated)
 * @param num_outputs Number of output tensors
 * @param stats Optional execution statistics
 * @return AGEE_OK on success
 */
int agee_execute_plan_with_tensors(agee_session_t session,
                                   agee_graph_plan_t plan, void **input_tensors,
                                   size_t num_inputs, void **output_tensors,
                                   size_t num_outputs,
                                   agee_exec_stats_t *stats);

/**
 * @brief Execute batched graphs in parallel
 *
 * @param session Session handle
 * @param plans Array of execution plans
 * @param num_plans Number of plans
 * @param stats_array Optional array of statistics (NULL = don't collect)
 * @return AGEE_OK on success
 */
int agee_execute_batch(agee_session_t session, agee_graph_plan_t *plans,
                       size_t num_plans, agee_exec_stats_t *stats_array);

/* ========================================================================== */
/* Memory Management */
/* ========================================================================== */

/**
 * @brief Estimate peak memory usage for a plan
 *
 * @param plan Execution plan
 * @param out_peak_bytes Output peak memory in bytes
 * @return AGEE_OK on success
 */
int agee_estimate_memory(agee_graph_plan_t plan, size_t *out_peak_bytes);

/**
 * @brief Pre-allocate memory for a plan
 *
 * Pre-allocates all required memory to avoid allocation overhead
 * during execution.
 *
 * @param session Session handle
 * @param plan Execution plan
 * @return AGEE_OK on success
 */
int agee_preallocate_memory(agee_session_t session, agee_graph_plan_t plan);

/* ========================================================================== */
/* Query & Inspection */
/* ========================================================================== */

/**
 * @brief Get plan information
 *
 * @param plan Plan handle
 * @param out_info Output string (caller must free with agee_free_string)
 * @return AGEE_OK on success
 */
int agee_get_plan_info(agee_graph_plan_t plan, char **out_info);

/**
 * @brief Get session statistics
 *
 * @param session Session handle
 * @param out_stats Output statistics structure
 * @return AGEE_OK on success
 */
int agee_get_session_stats(agee_session_t session,
                           agee_exec_stats_t *out_stats);

/**
 * @brief Get system information
 *
 * @param out_info Output string (caller must free with agee_free_string)
 * @return AGEE_OK on success
 */
int agee_get_system_info(char **out_info);

/**
 * @brief Free a string allocated by AGEE
 *
 * @param str String to free
 */
void agee_free_string(char *str);

/* ========================================================================== */
/* Utility Functions */
/* ========================================================================== */

/**
 * @brief Convert error code to string
 *
 * @param error Error code
 * @return Error message string
 */
const char *agee_strerror(int error);

/**
 * @brief Export execution plan to DOT format (for visualization)
 *
 * @param plan Plan handle
 * @param filename Output file path
 * @return AGEE_OK on success
 */
int agee_export_plan_dot(agee_graph_plan_t plan, const char *filename);

/**
 * @brief Run comprehensive self-test
 *
 * Tests all components and validates functionality.
 *
 * @param verbose Print detailed results (1/0)
 * @return AGEE_OK if all tests pass
 */
int agee_self_test(int verbose);

/**
 * @brief Benchmark graph execution with/without optimizations
 *
 * @param session Session handle
 * @param plan Plan to benchmark
 * @param iterations Number of iterations
 * @return AGEE_OK on success
 */
int agee_benchmark_plan(agee_session_t session, agee_graph_plan_t plan,
                        int iterations);

#ifdef __cplusplus
}
#endif

#endif /* JCORE_ADAPTIVE_GRAPH_EXECUTION_ENGINE_H_ */