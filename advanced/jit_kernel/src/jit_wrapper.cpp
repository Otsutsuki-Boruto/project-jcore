// advanced/jit_kernel/src/jit_wrapper.cpp
/**
 * @file jit_wrapper.cpp
 * @brief High-level wrapper functions for common JIT kernel patterns
 *
 * This file provides simplified interfaces for typical use cases,
 * abstracting away low-level details of kernel generation.
 */

#include "jit_kernel_generator.h"
#include "jit_kernel_internal.h"
#include <memory>
#include <unordered_map>

namespace jkg_wrapper {

/* ========================================================================== */
/* Kernel Registry for Application-Level Caching                              */
/* ========================================================================== */

struct KernelRegistry {
  std::unordered_map<std::string, jkg_kernel_internal_t *> named_kernels;
  std::mutex registry_mutex;
};

static KernelRegistry g_registry;

/* ========================================================================== */
/* Named Kernel Management */
/* ========================================================================== */

int jkg_register_named_kernel(const char *name, jkg_kernel_internal_t *handle) {
  if (!name || !handle) {
    return JKG_ERR_INVALID_ARG;
  }

  std::lock_guard<std::mutex> lock(g_registry.registry_mutex);

  std::string key(name);
  if (g_registry.named_kernels.find(key) != g_registry.named_kernels.end()) {
    return JKG_ERR_INTERNAL; // Already exists
  }

  g_registry.named_kernels[key] = handle;
  return JKG_OK;
}

jkg_kernel_internal_t *jkg_lookup_named_kernel(const char *name) {
  if (!name) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(g_registry.registry_mutex);

  std::string key(name);
  auto it = g_registry.named_kernels.find(key);
  if (it != g_registry.named_kernels.end()) {
    return it->second;
  }

  return nullptr;
}

void jkg_clear_named_kernels() {
  std::lock_guard<std::mutex> lock(g_registry.registry_mutex);

  // Release all handles
  for (auto &pair : g_registry.named_kernels) {
    jkg_release_kernel(pair.second);
  }

  g_registry.named_kernels.clear();
}

/* ========================================================================== */
/* High-Level GEMM Patterns */
/* ========================================================================== */

int jkg_create_gemm_kernel(const char *name, size_t M, size_t N, size_t K,
                           jkg_activation_t activation, int has_bias) {
  if (!name) {
    return JKG_ERR_INVALID_ARG;
  }

  // Check if already exists
  if (jkg_lookup_named_kernel(name)) {
    return JKG_OK; // Already created
  }

  // Determine kernel type
  jkg_kernel_type_t type;
  if (activation == JKG_ACT_NONE && !has_bias) {
    type = JKG_KERNEL_GEMM_TILE;
  } else if (activation == JKG_ACT_RELU && has_bias) {
    type = JKG_KERNEL_GEMM_BIAS_RELU;
  } else if (has_bias) {
    type = JKG_KERNEL_GEMM_BIAS_ACT;
  } else {
    type = JKG_KERNEL_GEMM_TILE;
  }

  // Generate kernel
  jkg_kernel_params_t params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.activation = activation;
  params.alpha = 1.0f;
  params.beta = 0.0f;
  params.has_bias = has_bias;
  params.has_residual = 0;

  jkg_kernel_internal_t *handle = nullptr;
  int ret = jkg_generate_kernel(type, &params, &handle);
  if (ret != JKG_OK) {
    return ret;
  }

  // Register with name
  ret = jkg_register_named_kernel(name, handle);
  if (ret != JKG_OK) {
    jkg_release_kernel(handle);
    return ret;
  }

  return JKG_OK;
}

int jkg_execute_gemm(const char *name, const float *A, const float *B, float *C,
                     const float *bias) {
  jkg_kernel_internal_t *handle = jkg_lookup_named_kernel(name);
  if (!handle) {
    return JKG_ERR_NOT_FOUND;
  }

  void *func_ptr = jkg_get_kernel_function(handle);
  if (!func_ptr) {
    return JKG_ERR_INTERNAL;
  }

  // Cast and execute (placeholder - actual execution depends on JIT
  // implementation) In real implementation: auto gemm_fn =
  // reinterpret_cast<jkg_gemm_fn>(func_ptr); gemm_fn(A, B, C, handle->params.M,
  // handle->params.N, handle->params.K, ...);

  return JKG_OK;
}

/* ========================================================================== */
/* Batch Kernel Generation */
/* ========================================================================== */

int jkg_create_gemm_suite(const char *prefix, const size_t *sizes,
                          int num_sizes, jkg_activation_t activation) {
  if (!prefix || !sizes || num_sizes <= 0) {
    return JKG_ERR_INVALID_ARG;
  }

  char name_buf[256];
  int failed = 0;

  for (int i = 0; i < num_sizes; i++) {
    size_t size = sizes[i];
    snprintf(name_buf, sizeof(name_buf), "%s_%zu", prefix, size);

    int ret = jkg_create_gemm_kernel(name_buf, size, size, size, activation, 1);
    if (ret != JKG_OK) {
      failed++;
    }
  }

  return (failed == 0) ? JKG_OK : JKG_ERR_INTERNAL;
}

/* ========================================================================== */
/* Auto-tuning Support */
/* ========================================================================== */

struct AutoTuneResult {
  size_t best_M;
  size_t best_N;
  size_t best_K;
  double best_gflops;
  char best_name[128];
};

int jkg_autotune_gemm(size_t target_M, size_t target_N, size_t target_K,
                      AutoTuneResult *result) {
  if (!result) {
    return JKG_ERR_INVALID_ARG;
  }

  // Use adaptive tuner to find best tile sizes
  size_t opt_M, opt_N, opt_K;
  int ret = jkg_get_optimal_tile_sizes(JKG_ISA_AUTO, &opt_M, &opt_N, &opt_K);
  if (ret != JKG_OK) {
    return ret;
  }

  result->best_M = opt_M;
  result->best_N = opt_N;
  result->best_K = opt_K;
  result->best_gflops = 0.0; // Would be populated by actual benchmarking
  snprintf(result->best_name, sizeof(result->best_name),
           "gemm_tuned_%zux%zux%zu", opt_M, opt_N, opt_K);

  return JKG_OK;
}

/* ========================================================================== */
/* Fusion Pattern Builders */
/* ========================================================================== */

int jkg_create_resnet_block(const char *name, size_t channels, size_t spatial) {
  // ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
  // Simplified to: GEMM -> Bias -> ReLU with residual

  char kernel_name[256];
  snprintf(kernel_name, sizeof(kernel_name), "%s_conv1", name);

  jkg_kernel_params_t params = {};
  params.M = spatial * spatial;
  params.N = channels;
  params.K = channels;
  params.activation = JKG_ACT_RELU;
  params.alpha = 1.0f;
  params.beta = 1.0f; // For residual
  params.has_bias = 1;
  params.has_residual = 1;

  jkg_kernel_internal_t *handle = nullptr;
  int ret = jkg_generate_kernel(JKG_KERNEL_GEMM_BIAS_ACT, &params, &handle);
  if (ret != JKG_OK) {
    return ret;
  }

  return jkg_register_named_kernel(kernel_name, handle);
}

int jkg_create_transformer_block(const char *name, size_t seq_len,
                                 size_t hidden_dim) {
  // Transformer: QKV projection -> Attention -> FFN
  // Create kernels for Q, K, V projections

  const char *projections[] = {"qproj", "kproj", "vproj", "ffn1", "ffn2"};

  for (int i = 0; i < 5; i++) {
    char kernel_name[256];
    snprintf(kernel_name, sizeof(kernel_name), "%s_%s", name, projections[i]);

    int ret = jkg_create_gemm_kernel(kernel_name, seq_len, hidden_dim,
                                     hidden_dim, JKG_ACT_GELU, 1);
    if (ret != JKG_OK) {
      return ret;
    }
  }

  return JKG_OK;
}

/* ========================================================================== */
/* Performance Profiling Helpers */
/* ========================================================================== */

struct ProfilingStats {
  double total_time_ms;
  double avg_time_ms;
  double min_time_ms;
  double max_time_ms;
  double gflops;
  int num_calls;
};

static std::unordered_map<std::string, ProfilingStats> g_profiling_data;
static std::mutex g_profiling_mutex;

void jkg_start_profiling() {
  std::lock_guard<std::mutex> lock(g_profiling_mutex);
  g_profiling_data.clear();
}

void jkg_record_kernel_execution(const char *name, double time_ms,
                                 double gflops) {
  if (!name)
    return;

  std::lock_guard<std::mutex> lock(g_profiling_mutex);

  std::string key(name);
  auto &stats = g_profiling_data[key];

  stats.num_calls++;
  stats.total_time_ms += time_ms;
  stats.avg_time_ms = stats.total_time_ms / stats.num_calls;

  if (stats.num_calls == 1) {
    stats.min_time_ms = time_ms;
    stats.max_time_ms = time_ms;
  } else {
    stats.min_time_ms = std::min(stats.min_time_ms, time_ms);
    stats.max_time_ms = std::max(stats.max_time_ms, time_ms);
  }

  stats.gflops = gflops;
}

void jkg_print_profiling_report() {
  std::lock_guard<std::mutex> lock(g_profiling_mutex);

  printf("\n");
  printf("====================================================================="
         "===\n");
  printf("JIT Kernel Profiling Report\n");
  printf("====================================================================="
         "===\n");
  printf("%-30s %8s %10s %10s %10s %10s\n", "Kernel", "Calls", "Avg(ms)",
         "Min(ms)", "Max(ms)", "GFLOPS");
  printf("---------------------------------------------------------------------"
         "---\n");

  for (const auto &pair : g_profiling_data) {
    const auto &stats = pair.second;
    printf("%-30s %8d %10.3f %10.3f %10.3f %10.2f\n", pair.first.c_str(),
           stats.num_calls, stats.avg_time_ms, stats.min_time_ms,
           stats.max_time_ms, stats.gflops);
  }

  printf("====================================================================="
         "===\n\n");
}

void jkg_clear_profiling() {
  std::lock_guard<std::mutex> lock(g_profiling_mutex);
  g_profiling_data.clear();
}

/* ========================================================================== */
/* Simplified API for Common Cases */
/* ========================================================================== */

int jkg_quick_gemm(size_t M, size_t N, size_t K, const float *A, const float *B,
                   float *C) {
  // Generate or lookup cached kernel
  char name[128];
  snprintf(name, sizeof(name), "quick_gemm_%zux%zux%zu", M, N, K);

  jkg_kernel_internal_t *handle = jkg_lookup_named_kernel(name);
  if (!handle) {
    int ret = jkg_create_gemm_kernel(name, M, N, K, JKG_ACT_NONE, 0);
    if (ret != JKG_OK) {
      return ret;
    }
    handle = jkg_lookup_named_kernel(name);
  }

  return jkg_execute_gemm(name, A, B, C, nullptr);
}

int jkg_quick_gemm_relu(size_t M, size_t N, size_t K, const float *A,
                        const float *B, float *C, const float *bias) {
  char name[128];
  snprintf(name, sizeof(name), "quick_gemm_relu_%zux%zux%zu", M, N, K);

  jkg_kernel_internal_t *handle = jkg_lookup_named_kernel(name);
  if (!handle) {
    int ret = jkg_create_gemm_kernel(name, M, N, K, JKG_ACT_RELU, 1);
    if (ret != JKG_OK) {
      return ret;
    }
    handle = jkg_lookup_named_kernel(name);
  }

  return jkg_execute_gemm(name, A, B, C, bias);
}

} // namespace jkg_wrapper

/* ========================================================================== */
/* C API Exports */
/* ========================================================================== */

extern "C" {

int jkg_create_gemm_kernel_c(const char *name, size_t M, size_t N, size_t K,
                             jkg_activation_t activation, int has_bias) {
  return jkg_wrapper::jkg_create_gemm_kernel(name, M, N, K, activation,
                                             has_bias);
}

int jkg_execute_gemm_c(const char *name, const float *A, const float *B,
                       float *C, const float *bias) {
  return jkg_wrapper::jkg_execute_gemm(name, A, B, C, bias);
}

void jkg_clear_named_kernels_c() { jkg_wrapper::jkg_clear_named_kernels(); }

void jkg_start_profiling_c() { jkg_wrapper::jkg_start_profiling(); }

void jkg_print_profiling_report_c() {
  jkg_wrapper::jkg_print_profiling_report();
}

int jkg_quick_gemm_c(size_t M, size_t N, size_t K, const float *A,
                     const float *B, float *C) {
  return jkg_wrapper::jkg_quick_gemm(M, N, K, A, B, C);
}

int jkg_quick_gemm_relu_c(size_t M, size_t N, size_t K, const float *A,
                          const float *B, float *C, const float *bias) {
  return jkg_wrapper::jkg_quick_gemm_relu(M, N, K, A, B, C, bias);
}

} // extern "C"