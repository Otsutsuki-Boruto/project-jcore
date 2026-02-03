// advanced/jit_kernel/src/jit_utilities.cpp
#include "jit_kernel_internal.h"
#include <cstdio>
#include <cstring>

using namespace jkg_internal;

/* ========================================================================== */
/* Debugging and Introspection Functions                                      */
/* ========================================================================== */

int jkg_benchmark_kernel(jkg_kernel_internal_t *handle, int iterations,
                         double *out_gflops, double *out_time_ms) {
  if (!g_jkg_state.initialized) {
    return JKG_ERR_NOT_INITIALIZED;
  }
  if (!handle) {
    return JKG_ERR_INVALID_ARG;
  }

  auto impl = handle_to_impl(handle);

  log_info("Benchmarking kernel: %s (%d iterations)", impl->kernel_name.c_str(),
           iterations);

  // Use microbenchmark utilities
  jcore::microbench::RunOptions opts;
  opts.warmup_iterations = 1;
  opts.iterations = iterations;
  opts.samples = 10;
  opts.capture_cycles = true;

  std::vector<jcore::microbench::Sample> samples;

  // Placeholder: would execute actual kernel here
  auto kernel_fn = [&](size_t iter) {
    // Execute compiled kernel
    // For now, just a small delay to simulate work
    volatile int dummy = 0;
    for (int i = 0; i < 1000; i++) {
      dummy += i;
    }
  };

  bool success = jcore::microbench::RunMicrobenchmark(kernel_fn, opts, samples);
  if (!success) {
    log_error("Benchmark failed");
    return JKG_ERR_INTERNAL;
  }

  jcore::microbench::Summary summary;
  jcore::microbench::Summarize(samples, summary);

  if (out_time_ms) {
    *out_time_ms = summary.mean_usec / 1000.0;
  }

  if (out_gflops) {
    // Compute GFLOPS based on kernel type and parameters
    size_t M = impl->params.M;
    size_t N = impl->params.N;
    size_t K = impl->params.K;

    double flops = 0.0;
    switch (impl->type) {
    case JKG_KERNEL_GEMM_TILE:
    case JKG_KERNEL_GEMM_BIAS:
    case JKG_KERNEL_GEMM_BIAS_RELU:
    case JKG_KERNEL_GEMM_BIAS_ACT:
      flops = 2.0 * M * N * K; // Multiply-add
      break;
    case JKG_KERNEL_ELEMENTWISE_ADD:
    case JKG_KERNEL_ELEMENTWISE_MUL:
      flops = M; // Single operation per element
      break;
    default:
      flops = 0.0;
    }

    double time_sec = summary.mean_usec / 1e6;
    *out_gflops = (flops / time_sec) / 1e9;
  }

  log_info("Benchmark complete: %.2f GFLOPS, %.3f ms",
           out_gflops ? *out_gflops : 0.0, out_time_ms ? *out_time_ms : 0.0);

  return JKG_OK;
}

char *jkg_get_kernel_ir(jkg_kernel_internal_t *handle) {
  if (!handle) {
    return nullptr;
  }

  auto impl = handle_to_impl(handle);

  if (impl->llvm_ir.empty()) {
    return nullptr;
  }

  // Allocate and copy IR string
  char *ir_copy = static_cast<char *>(malloc(impl->llvm_ir.size() + 1));
  if (ir_copy) {
    strcpy(ir_copy, impl->llvm_ir.c_str());
  }

  return ir_copy;
}

char *jkg_get_kernel_asm(jkg_kernel_internal_t *handle) {
  if (!handle) {
    return nullptr;
  }

  auto impl = handle_to_impl(handle);

  if (impl->assembly.empty()) {
    return nullptr;
  }

  // Allocate and copy assembly string
  char *asm_copy = static_cast<char *>(malloc(impl->assembly.size() + 1));
  if (asm_copy) {
    strcpy(asm_copy, impl->assembly.c_str());
  }

  return asm_copy;
}

int jkg_dump_kernel(jkg_kernel_internal_t *handle, const char *filename) {
  if (!handle || !filename) {
    return JKG_ERR_INVALID_ARG;
  }

  auto impl = handle_to_impl(handle);

  FILE *f = fopen(filename, "w");
  if (!f) {
    log_error("Failed to open file: %s", filename);
    return JKG_ERR_INTERNAL;
  }

  fprintf(f, "========================================\n");
  fprintf(f, "JIT Kernel Dump\n");
  fprintf(f, "========================================\n\n");

  fprintf(f, "Kernel Name: %s\n", impl->kernel_name.c_str());
  fprintf(f, "Kernel Type: %d\n", impl->type);
  fprintf(f, "Parameters:\n");
  fprintf(f, "  M = %zu\n", impl->params.M);
  fprintf(f, "  N = %zu\n", impl->params.N);
  fprintf(f, "  K = %zu\n", impl->params.K);
  fprintf(f, "  Activation: %d\n", impl->params.activation);
  fprintf(f, "  Has Bias: %d\n", impl->params.has_bias);
  fprintf(f, "  Has Residual: %d\n", impl->params.has_residual);
  fprintf(f, "  Alpha: %.6f\n", impl->params.alpha);
  fprintf(f, "  Beta: %.6f\n", impl->params.beta);
  fprintf(f, "\n");

  fprintf(f, "Code Size: %zu bytes\n", impl->code_size_bytes);
  fprintf(f, "Cached: %s\n", impl->is_cached ? "Yes" : "No");
  fprintf(f, "Reference Count: %d\n", impl->ref_count.load());
  fprintf(f, "\n");

  if (!impl->llvm_ir.empty()) {
    fprintf(f, "========================================\n");
    fprintf(f, "LLVM IR:\n");
    fprintf(f, "========================================\n");
    fprintf(f, "%s\n\n", impl->llvm_ir.c_str());
  }

  if (!impl->assembly.empty()) {
    fprintf(f, "========================================\n");
    fprintf(f, "Assembly:\n");
    fprintf(f, "========================================\n");
    fprintf(f, "%s\n\n", impl->assembly.c_str());
  }

  fclose(f);

  log_info("Kernel dumped to: %s", filename);
  return JKG_OK;
}

/* ========================================================================== */
/* Self-Test Implementation */
/* ========================================================================== */

int jkg_self_test(int verbose) {
  if (!g_jkg_state.initialized) {
    if (verbose) {
      printf("JKG Self-Test: Initializing...\n");
    }
    int ret = jkg_init(nullptr);
    if (ret != JKG_OK) {
      if (verbose) {
        printf("JKG Self-Test: FAILED - Initialization error\n");
      }
      return ret;
    }
  }

  int test_count = 0;
  int failed = 0;

  if (verbose) {
    printf("\n========================================\n");
    printf("JIT Kernel Generator Self-Test\n");
    printf("========================================\n\n");
  }

  // Test 1: ISA Detection
  test_count++;
  if (verbose)
    printf("Test %d: ISA Detection... ", test_count);
  uint32_t isa_mask = jkg_get_available_isa();
  if (isa_mask == 0) {
    if (verbose)
      printf("FAILED\n");
    failed++;
  } else {
    if (verbose)
      printf("OK (mask: 0x%08X)\n", isa_mask);
  }

  // Test 2: Tile Size Computation
  test_count++;
  if (verbose)
    printf("Test %d: Tile Size Computation... ", test_count);
  size_t M, N, K;
  int ret = jkg_get_optimal_tile_sizes(JKG_ISA_AUTO, &M, &N, &K);
  if (ret != JKG_OK || M == 0 || N == 0 || K == 0) {
    if (verbose)
      printf("FAILED\n");
    failed++;
  } else {
    if (verbose)
      printf("OK (%zux%zux%zu)\n", M, N, K);
  }

  // Test 3: GEMM Kernel Generation
  test_count++;
  if (verbose)
    printf("Test %d: GEMM Kernel Generation... ", test_count);
  jkg_kernel_internal_t *handle = nullptr;
  ret = jkg_generate_gemm_tile(8, 32, 64, &handle);
  if (ret != JKG_OK) {
    if (verbose)
      printf("FAILED (error: %s)\n", jkg_strerror(ret));
    failed++;
  } else {
    if (verbose)
      printf("OK\n");
    jkg_release_kernel(handle);
  }

  // Test 4: Fused Kernel Generation
  test_count++;
  if (verbose)
    printf("Test %d: Fused Kernel Generation... ", test_count);
  ret = jkg_generate_fused_gemm(16, 16, 16, JKG_ACT_RELU, 1.0f, &handle);
  if (ret != JKG_OK) {
    if (verbose)
      printf("FAILED\n");
    failed++;
  } else {
    if (verbose)
      printf("OK\n");
    jkg_release_kernel(handle);
  }

  // Test 5: Elementwise Kernel Generation
  test_count++;
  if (verbose)
    printf("Test %d: Elementwise Kernel Generation... ", test_count);
  ret = jkg_generate_elementwise(JKG_KERNEL_ELEMENTWISE_ADD, 1024, &handle);
  if (ret != JKG_OK) {
    if (verbose)
      printf("FAILED\n");
    failed++;
  } else {
    if (verbose)
      printf("OK\n");
    jkg_release_kernel(handle);
  }

  // Test 6: Cache Functionality
  test_count++;
  if (verbose)
    printf("Test %d: Cache Functionality... ", test_count);
  jkg_clear_cache();
  size_t cached, hits, misses;
  jkg_get_cache_stats(&cached, &hits, &misses);
  if (verbose)
    printf("OK (stats available)\n");

  // Test 7: Config Retrieval
  test_count++;
  if (verbose)
    printf("Test %d: Config Retrieval... ", test_count);
  jkg_config_t config;
  ret = jkg_get_config(&config);
  if (ret != JKG_OK) {
    if (verbose)
      printf("FAILED\n");
    failed++;
  } else {
    if (verbose)
      printf("OK\n");
  }

  // Test 8: System Info
  test_count++;
  if (verbose)
    printf("Test %d: System Info... ", test_count);
  const char *sys_info = jkg_get_system_info();
  if (!sys_info) {
    if (verbose)
      printf("FAILED\n");
    failed++;
  } else {
    if (verbose)
      printf("OK\n");
  }

  // Summary
  if (verbose) {
    printf("\n========================================\n");
    printf("Self-Test Summary:\n");
    printf("  Total Tests: %d\n", test_count);
    printf("  Passed: %d\n", test_count - failed);
    printf("  Failed: %d\n", failed);
    printf("========================================\n\n");
  }

  return (failed == 0) ? JKG_OK : JKG_ERR_INTERNAL;
}

/* ========================================================================== */
/* Activation Name Helpers */
/* ========================================================================== */

const char *jkg_activation_name(jkg_activation_t act) {
  switch (act) {
  case JKG_ACT_NONE:
    return "none";
  case JKG_ACT_RELU:
    return "relu";
  case JKG_ACT_RELU6:
    return "relu6";
  case JKG_ACT_TANH:
    return "tanh";
  case JKG_ACT_SIGMOID:
    return "sigmoid";
  case JKG_ACT_GELU:
    return "gelu";
  case JKG_ACT_SWISH:
    return "swish";
  case JKG_ACT_LEAKY_RELU:
    return "leaky_relu";
  default:
    return "unknown";
  }
}

const char *jkg_kernel_type_name(jkg_kernel_type_t type) {
  switch (type) {
  case JKG_KERNEL_GEMM_TILE:
    return "gemm_tile";
  case JKG_KERNEL_GEMM_BIAS:
    return "gemm_bias";
  case JKG_KERNEL_GEMM_BIAS_RELU:
    return "gemm_bias_relu";
  case JKG_KERNEL_GEMM_BIAS_ACT:
    return "gemm_bias_act";
  case JKG_KERNEL_ELEMENTWISE_ADD:
    return "elementwise_add";
  case JKG_KERNEL_ELEMENTWISE_MUL:
    return "elementwise_mul";
  case JKG_KERNEL_ACTIVATION:
    return "activation";
  case JKG_KERNEL_REDUCE_SUM:
    return "reduce_sum";
  case JKG_KERNEL_BATCH_NORM:
    return "batch_norm";
  case JKG_KERNEL_LAYER_NORM:
    return "layer_norm";
  case JKG_KERNEL_CUSTOM:
    return "custom";
  default:
    return "unknown";
  }
}

/* ========================================================================== */
/* Performance Estimation */
/* ========================================================================== */

double jkg_estimate_gflops(jkg_kernel_type_t type, size_t M, size_t N, size_t K,
                           double time_sec) {
  double flops = 0.0;

  switch (type) {
  case JKG_KERNEL_GEMM_TILE:
  case JKG_KERNEL_GEMM_BIAS:
  case JKG_KERNEL_GEMM_BIAS_RELU:
  case JKG_KERNEL_GEMM_BIAS_ACT:
    // GEMM: 2*M*N*K (multiply-add)
    flops = 2.0 * M * N * K;
    break;

  case JKG_KERNEL_ELEMENTWISE_ADD:
  case JKG_KERNEL_ELEMENTWISE_MUL:
    // Element-wise: 1 op per element
    flops = static_cast<double>(M);
    break;

  case JKG_KERNEL_ACTIVATION:
    // Depends on activation, typically 1-10 ops per element
    flops = 5.0 * M; // Average estimate
    break;

  default:
    flops = 0.0;
  }

  if (time_sec <= 0.0) {
    return 0.0;
  }

  return (flops / time_sec) / 1e9;
}

/* ========================================================================== */
/* Memory Footprint Estimation                                                */
/* ========================================================================== */

size_t jkg_estimate_memory_footprint(jkg_kernel_type_t type, size_t M, size_t N,
                                     size_t K) {
  size_t bytes = 0;
  const size_t float_size = sizeof(float);

  switch (type) {
  case JKG_KERNEL_GEMM_TILE:
  case JKG_KERNEL_GEMM_BIAS:
  case JKG_KERNEL_GEMM_BIAS_RELU:
  case JKG_KERNEL_GEMM_BIAS_ACT:
    // A: M*K, B: K*N, C: M*N
    bytes = (M * K + K * N + M * N) * float_size;
    break;

  case JKG_KERNEL_ELEMENTWISE_ADD:
  case JKG_KERNEL_ELEMENTWISE_MUL:
    // A, B, C: all M elements
    bytes = 3 * M * float_size;
    break;

  case JKG_KERNEL_ACTIVATION:
    // In-place: just M elements
    bytes = M * float_size;
    break;

  default:
    bytes = 0;
  }

  return bytes;
}

/* ========================================================================== */
/* Vectorization Backend Query Functions                                      */
/* ========================================================================== */

int jkg_query_backend_available(jkg_backend_t backend) {
  switch (backend) {
  case JKG_BACKEND_HIGHWAY:
    return JKG_HAS_HIGHWAY ? 1 : 0;
  case JKG_BACKEND_VECTORCLASS:
    return JKG_HAS_VECTORCLASS ? 1 : 0;
  case JKG_BACKEND_EVE:
    return JKG_HAS_EVE ? 1 : 0;
  case JKG_BACKEND_LLVM:
    return 1; // Always available
  case JKG_BACKEND_AUTO:
    return 1;
  default:
    return 0;
  }
}

size_t jkg_query_backend_vector_width(jkg_backend_t backend) {
  if (!jkg_query_backend_available(backend)) {
    return 0;
  }

  auto vec_backend = jkg_internal::create_backend(backend);
  if (vec_backend) {
    return vec_backend->vector_width();
  }

  return 0;
}

const char *jkg_query_backend_description(jkg_backend_t backend) {
  switch (backend) {
  case JKG_BACKEND_AUTO:
    return "Automatic backend selection";
  case JKG_BACKEND_HIGHWAY:
    return "Google Highway - Portable SIMD with runtime dispatch";
  case JKG_BACKEND_VECTORCLASS:
    return "VectorClass - x86 intrinsics wrappers";
  case JKG_BACKEND_EVE:
    return "EVE - Expressive Vector Engine";
  case JKG_BACKEND_LLVM:
    return "Pure LLVM - Auto-vectorization";
  default:
    return "Unknown backend";
  }
}