// advanced/operator_graph/src/operator_graph_core.cpp
/**
 * @file operator_graph_core.cpp
 * @brief Core initialization and state management for Operator Graph Runtime
 */

#include "operator_graph_internal.h"
#include "pool_manager.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace og_internal {
// Global state instance
OGRuntimeState g_og_state;
} // namespace og_internal

using namespace og_internal;

/* ========================================================================== */
/* Initialization & Shutdown                                                  */
/* ========================================================================== */

int og_init(const og_config_t *config) {
  std::lock_guard<std::mutex> lock(g_og_state.state_mutex);

  if (g_og_state.initialized) {
    return OG_OK; // Already initialized
  }

  // Set default configuration
  if (config) {
    g_og_state.config = *config;
  } else {
    // Default configuration
    g_og_state.config.enable_fusion = 1;
    g_og_state.config.enable_parallelism = 1;
    g_og_state.config.enable_memory_reuse = 1;
    g_og_state.config.enable_pattern_matching = 1;
    g_og_state.config.max_fusion_depth = 8;
    g_og_state.config.num_threads = 0; // Auto-detect
    g_og_state.config.verbose = 0;
    g_og_state.config.fusion_threshold = 1.2; // minimum speedup
  }

  // Initialize Kernel Fusion Engine
  kfe_config_t kfe_cfg = {};
  kfe_cfg.num_threads = g_og_state.config.num_threads;
  kfe_cfg.enable_vectorization = 1;
  kfe_cfg.enable_cache_blocking = 1;
  kfe_cfg.enable_prefetch = 1;
  kfe_cfg.enable_kernel_autotuning = 1;
  kfe_cfg.workspace_size_mb = 256; // 256 MB workspace
  kfe_cfg.verbose = g_og_state.config.verbose;

  int ret = kfe_init(&kfe_cfg);
  if (ret != KFE_OK) {
    if (g_og_state.config.verbose) {
      std::cerr << "[OG] Failed to initialize Kernel Fusion Engine: "
                << kfe_strerror(ret) << std::endl;
    }
    return OG_ERR_INTERNAL;
  }
  std::cerr << "[OG] Kernel Fusion Engine Initialized" << std :: endl;

  // Detect CPU information
  g_og_state.cpu_info = detect_cpu_info();

  // Detect Cache information
  g_og_state.cache_info = ffm_cache_init();

  // Initialize memory pool manager for tensor allocations
  size_t pool_size = 512 * 1024 * 1024; // 512 MB
  size_t chunk_size = 1 * 1024 * 1024; // 1 MB chunks for tensors
  pm_status_t pm_status = pm_init(&g_og_state.pool_manager, pool_size, chunk_size, 0, -1);
  if (pm_status != PM_OK) {
    if (g_og_state.config.verbose) {
      std::cerr << "[OG] Warning: Pool manager Init failed, using direct allocation\n";
    }
    g_og_state.pool_manager = nullptr;
  }

  if (g_og_state.config.verbose) {
    std::cout << "[OG] Operator Graph Runtime Initialized Successfully"
              << std::endl;
    std::cout << "[OG] CPU: " << g_og_state.cpu_info.cores << " cores, "
              << "L1D: " << g_og_state.cpu_info.l1d_kb << " KB, "
              << "L2: " << g_og_state.cpu_info.l2_kb << " KB, "
              << "L3: " << g_og_state.cpu_info.l3_kb << " KB" << std::endl;
    std::cout << "[OG] Features: AVX=" << g_og_state.cpu_info.avx
              << " AVX2=" << g_og_state.cpu_info.avx2
              << " AVX512=" << g_og_state.cpu_info.avx512
              << " AMX=" << g_og_state.cpu_info.amx << std::endl;
    std::cout << "[OG] Fusion: "
              << (g_og_state.config.enable_fusion ? "ON" : "OFF")
              << ", Parallelism: "
              << (g_og_state.config.enable_parallelism ? "ON" : "OFF")
              << ", Pattern Matching: "
              << (g_og_state.config.enable_pattern_matching ? "ON" : "OFF")
              << std::endl;
  }

  g_og_state.initialized = true;
  return OG_OK;
}

void og_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_og_state.state_mutex);

  if (!g_og_state.initialized) {
    return;
  }

  if (g_og_state.config.verbose) {
    std::cout << "[OG] Shutting down Operator Graph Runtime" << std::endl;
    std::cout << "[OG] Statistics: " << std::endl;
    std::cout << "[OG]   Total graphs created: "
              << g_og_state.total_graphs_created.load() << std::endl;
    std::cout << "[OG]   Total graphs executed: "
              << g_og_state.total_graphs_executed.load() << std::endl;
    std::cout << "[OG]   Total fusion groups executed: "
              << g_og_state.total_fusion_groups_executed.load() << std::endl;
  }

  // Shutdown dependencies
  kfe_shutdown();

  // Cleanup pool manager
  if (g_og_state.pool_manager) {
    pm_shutdown(g_og_state.pool_manager);
    g_og_state.pool_manager = nullptr;
  }

  // Cleanup scheduler if allocated
  if (g_og_state.scheduler) {
    delete g_og_state.scheduler;
    g_og_state.scheduler = nullptr;
  }

  // *** ADD THESE 3 LINES: ***
  g_og_state.total_graphs_created.store(0, std::memory_order_release);
  g_og_state.total_graphs_executed.store(0, std::memory_order_release);
  g_og_state.total_fusion_groups_executed.store(0, std::memory_order_release);

  g_og_state.initialized = false;
}

int og_is_initialized(void) {
  std::lock_guard<std::mutex> lock(g_og_state.state_mutex);
  return g_og_state.initialized ? 1 : 0;
}

/* ========================================================================== */
/* Graph Lifecycle Management                                                 */
/* ========================================================================== */

int og_create_graph(og_graph_t *out_graph) {

  if (!out_graph) {
    return OG_ERR_INVALID_ARG;
  }

  if (!g_og_state.initialized) {
    return OG_ERR_NOT_INITIALIZED;
  }

  try {
    // CRITICAL FIX: Use calloc-style allocation to zero memory
    void* raw_mem = operator new(sizeof(GraphImpl));
    std::memset(raw_mem, 0, sizeof(GraphImpl));  // Zero out the memory!
    GraphImpl *graph = new (raw_mem) GraphImpl();  // Placement new


    *out_graph = reinterpret_cast<og_graph_t>(graph);

    g_og_state.total_graphs_created.fetch_add(1, std::memory_order_relaxed);

    if (g_og_state.config.verbose) {
      std::cout << "[OG] Created new graph: " << graph << std::endl;
    }

    return OG_OK;

  } catch (const std::bad_alloc &e) {
    return OG_ERR_NO_MEMORY;
  } catch (const std::exception &e) {
    return OG_ERR_INTERNAL;
  } catch (...) {
    return OG_ERR_INTERNAL;
  }
}

void og_destroy_graph(og_graph_t graph) {
  if (!graph) {
    return;
  }

  GraphImpl *impl = reinterpret_cast<GraphImpl *>(graph);

  if (g_og_state.config.verbose) {
    std::lock_guard<std::mutex> lock(impl->graph_mutex);
    std::cout << "[OG] Destroying graph: " << impl << std::endl;
    std::cout << "[OG]   Nodes: " << impl->nodes.size()
              << ", Tensors: " << impl->tensors.size()
              << ", Fusion groups: " << impl->fusion_groups.size()
              << ", Finalized: " << (impl->is_finalized ? "Yes" : "No")
              << std::endl;
  }

  // Free allocated tensor data (must be done before clearing containers)
  {
    std::lock_guard<std::mutex> lock(impl->graph_mutex);
    for (auto &tensor_pair : impl->tensors) {
      TensorImpl *tensor = tensor_pair.second.get();
      if (tensor && tensor->allocated_data) {
        FreeTensorData(tensor->allocated_data, tensor->allocated_size);
        tensor->allocated_data = nullptr;
        tensor->allocated_size = 0;
      }
    }
  }

  // CRITICAL: Explicitly clear all node data structures
  {
    std::lock_guard<std::mutex> lock(impl->graph_mutex);
    for (auto &node_pair : impl->nodes) {
      NodeImpl *node = node_pair.second.get();
      if (node) {
        node->successor_nodes.clear();
        node->predecessor_nodes.clear();
        node->input_tensor_ids.clear();
        node->output_tensor_ids.clear();
        node->in_degree = 0;
        node->is_executed = false;
      }
    }
  }


  // Safe cleanup: clear all containers even if graph was not finalized
  // This prevents segfaults when destroying partially initialized graphs
  {
    std::lock_guard<std::mutex> lock(impl->graph_mutex);

    // Clear in safe order
    impl->fusion_groups.clear();
    impl->topological_order.clear();
    impl->fusion_execution_order.clear();
    impl->nodes.clear();
    impl->tensors.clear();
  }
  delete impl;
}

/* ========================================================================== */
/* Reset Execution State                                                     */
/* ========================================================================== */

int og_reset_execution_state(og_graph_t graph) {
  if (!graph) {
    return OG_ERR_INVALID_ARG;
  }

  og_internal::GraphImpl *impl =
      reinterpret_cast<og_internal::GraphImpl *>(graph);
  std::lock_guard<std::mutex> lock(impl->graph_mutex);

  // Reset all node execution states
  for (auto &node_pair : impl->nodes) {
    node_pair.second->is_executed = false;
    node_pair.second->execution_time_ms = 0.0;
  }

  // Reset all fusion group execution states
  for (auto &group_pair : impl->fusion_groups) {
    group_pair.second->is_executed = false;
    group_pair.second->execution_time_ms = 0.0;
  }

  return OG_OK;
}

/* ========================================================================== */
/* Memory Management                                                          */
/* ========================================================================== */

namespace og_internal {

std::unordered_map<void*, bool> tensor_allocation_map;
std::mutex tensor_alloc_mutex;

void *AllocateTensorData(size_t size_bytes) {
    if (size_bytes == 0) {
      return nullptr;
    }

    void *ptr = nullptr;
    bool used_pool = false;

    // Try pool manager first if available
    if (g_og_state.pool_manager) {
      ptr = pm_alloc(g_og_state.pool_manager);
      if (ptr) {
        used_pool = true;
      }
    }

    // Fallback to aligned allocation
    if (!ptr) {
      int ret = posix_memalign(&ptr, 64, size_bytes);
      if (ret != 0 || !ptr) {
        return nullptr;
      }
      used_pool = false;
    }

    // Track allocation
    {
      std::lock_guard<std::mutex> lock(tensor_alloc_mutex);
      tensor_allocation_map[ptr] = used_pool;
    }

    // Initialize to zero
    std::memset(ptr, 0, size_bytes);
    return ptr;
}

  void FreeTensorData(void *ptr, size_t size_bytes) {
  (void)size_bytes;
  if (!ptr) {
    return;
  }

  bool used_pool = false;
  {
    std::lock_guard<std::mutex> lock(tensor_alloc_mutex);
    auto it = tensor_allocation_map.find(ptr);
    if (it != tensor_allocation_map.end()) {
      used_pool = it->second;
      tensor_allocation_map.erase(it);
    }
  }

  if (used_pool && g_og_state.pool_manager) {
    pm_free(g_og_state.pool_manager, ptr);
  } else {
    free(ptr);
  }
}

} // namespace og_internal