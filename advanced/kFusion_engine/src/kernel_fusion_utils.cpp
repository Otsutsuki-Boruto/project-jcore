// advanced/kFusion_engine/src/kernel_fusion_utils.cpp

#include "jcore_isa_dispatch.h"
#include "kernel_fusion_engine_internal.h"
#include "mem_wrapper.h"
#include "pool_manager.h"

using namespace kfe_internal;

extern "C"
{

  /* ========================================================================== */
  /* Utility Functions                                                           */
  /* ========================================================================== */

  size_t kfe_estimate_memory_savings(size_t m, size_t n, kfe_fusion_type_t fusion_type)
  {
    switch (fusion_type)
    {
    case KFE_FUSION_GEMM_BIAS:
      return m * n * sizeof(float);
    case KFE_FUSION_GEMM_BIAS_ACTIVATION:
      return 2 * m * n * sizeof(float);
    case KFE_FUSION_GEMM_ELEMENTWISE_ADD:
    case KFE_FUSION_GEMM_ELEMENTWISE_MUL:
      return m * n * sizeof(float);
    case KFE_FUSION_GEMM_RESIDUAL:
      return 3 * m * n * sizeof(float);
    default:
      return 0;
    }
  }

  const char *kfe_activation_name(kfe_activation_t activation)
  {
    switch (activation)
    {
    case KFE_ACTIVATION_NONE:
      return "None";
    case KFE_ACTIVATION_RELU:
      return "ReLU";
    case KFE_ACTIVATION_RELU6:
      return "ReLU6";
    case KFE_ACTIVATION_TANH:
      return "Tanh";
    case KFE_ACTIVATION_SIGMOID:
      return "Sigmoid";
    case KFE_ACTIVATION_GELU:
      return "GELU";
    case KFE_ACTIVATION_SWISH:
      return "Swish";
    case KFE_ACTIVATION_LEAKY_RELU:
      return "LeakyReLU";
    default:
      return "Unknown";
    }
  }

  const char *kfe_fusion_name(kfe_fusion_type_t fusion_type)
  {
    switch (fusion_type)
    {
    case KFE_FUSION_GEMM_BIAS:
      return "GEMM+Bias";
    case KFE_FUSION_GEMM_BIAS_ACTIVATION:
      return "GEMM+Bias+Activation";
    case KFE_FUSION_GEMM_ELEMENTWISE_ADD:
      return "GEMM+ElementwiseAdd";
    case KFE_FUSION_GEMM_ELEMENTWISE_MUL:
      return "GEMM+ElementwiseMul";
    case KFE_FUSION_GEMM_RESIDUAL:
      return "GEMM+Bias+Residual+Activation";
    default:
      return "Unknown";
    }
  }

  const char *kfe_strerror(int error)
  {
    switch (error)
    {
    case KFE_OK:
      return "Success";
    case KFE_ERR_NOT_INITIALIZED:
      return "KFE not initialized";
    case KFE_ERR_INVALID_ARG:
      return "Invalid argument";
    case KFE_ERR_NO_MEMORY:
      return "Out of memory";
    case KFE_ERR_INTERNAL:
      return "Internal error";
    case KFE_ERR_UNSUPPORTED:
      return "Unsupported operation";
    case KFE_ERR_ALLOCATION:
      return "Allocation failed";
    default:
      return "Unknown error";
    }
  }

  const char *kfe_get_system_info(void)
  {
    static char info_buf[1024];
    if (!g_kfe_state.initialized)
    {
      snprintf(info_buf, sizeof(info_buf), "KFE not initialized");
      return info_buf;
    }

    jcore_features_t features = jcore_get_host_features();
    const char *mil_info = mil_get_system_info();

    snprintf(info_buf, sizeof(info_buf),
             "Kernel Fusion Engine\n"
             "  Threads: %zu\n"
             "  Vectorization: %s\n"
             "  AVX2: %s\n"
             "  AVX512: %s\n"
             "  Total Fused Ops: %zu\n"
             "  Total Memory Saved: %.2f MB\n"
             "  Backend Info: %s",
             kfe_get_num_threads(),
             g_kfe_state.config.enable_vectorization ? "Enabled" : "Disabled",
             (features & JCORE_FEAT_AVX2) ? "Yes" : "No",
             (features & JCORE_FEAT_AVX512) ? "Yes" : "No",
             g_kfe_state.total_fused_ops.load(),
             g_kfe_state.total_memory_saved.load() / (1024.0 * 1024.0),
             mil_info);

    return info_buf;
  }

  int kfe_self_test(int verbose)
  {
    if (!g_kfe_state.initialized)
    {
      return KFE_ERR_NOT_INITIALIZED;
    }

    const size_t M = 128, N = 128, K = 128;
    int passed = 0, failed = 0;

    // Allocate test matrices using pool manager if available
    pm_t *pool = g_kfe_state.pool_manager;
    float *A = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(ffm_aligned_alloc(64, M * K * sizeof(float)));
    float *B = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(ffm_aligned_alloc(64, K * N * sizeof(float)));
    float *C = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(ffm_aligned_alloc(64, M * N * sizeof(float)));
    float *bias = pool ? static_cast<float *>(pm_alloc(pool)) : static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));

    if (!A || !B || !C || !bias)
    {
      if (pool) {
        if (A) pm_free(pool, A);
        if (B) pm_free(pool, B);
        if (C) pm_free(pool, C);
        if (bias) pm_free(pool, bias);
      } else {
        ffm_free(A);
        ffm_free(B);
        ffm_free(C);
        ffm_free(bias);
      }
      return KFE_ERR_NO_MEMORY;
    }

    // Initialize with test data
    for (size_t i = 0; i < M * K; ++i)
      A[i] = static_cast<float>(i % 100) / 100.0f;
    for (size_t i = 0; i < K * N; ++i)
      B[i] = static_cast<float>(i % 100) / 100.0f;
    for (size_t i = 0; i < N; ++i)
      bias[i] = 0.1f;

    // Test 1: GEMM + Bias
    if (verbose)
      fprintf(stderr, "Testing GEMM+Bias... ");
    kfe_perf_stats_t stats = {};
    int ret = kfe_sgemm_bias(KFE_LAYOUT_ROW_MAJOR, KFE_NO_TRANS, KFE_NO_TRANS,
                             M, N, K, 1.0f, A, K, B, N, bias, C, N, &stats);
    if (ret == KFE_OK)
    {
      passed++;
      if (verbose)
        fprintf(stderr, "PASSED (%.2f GFLOPS)\n", stats.gflops);
    }
    else
    {
      failed++;
      if (verbose)
        fprintf(stderr, "FAILED\n");
    }

    // Test 2: GEMM + Bias + ReLU
    if (verbose)
      fprintf(stderr, "Testing GEMM+Bias+ReLU... ");
    ret = kfe_sgemm_bias_activation(KFE_LAYOUT_ROW_MAJOR, KFE_NO_TRANS, KFE_NO_TRANS,
                                    M, N, K, 1.0f, A, K, B, N, bias,
                                    KFE_ACTIVATION_RELU, C, N, &stats);
    if (ret == KFE_OK)
    {
      passed++;
      if (verbose)
        fprintf(stderr, "PASSED (%.2f GFLOPS)\n", stats.gflops);
    }
    else
    {
      failed++;
      if (verbose)
        fprintf(stderr, "FAILED\n");
    }

    // Test 3: Different activations
    kfe_activation_t activations[] = {
        KFE_ACTIVATION_SIGMOID, KFE_ACTIVATION_TANH, KFE_ACTIVATION_GELU};
    for (int i = 0; i < 3; ++i)
    {
      if (verbose)
        fprintf(stderr, "Testing GEMM+Bias+%s... ",
                kfe_activation_name(activations[i]));
      ret = kfe_sgemm_bias_activation(KFE_LAYOUT_ROW_MAJOR, KFE_NO_TRANS, KFE_NO_TRANS,
                                      M, N, K, 1.0f, A, K, B, N, bias,
                                      activations[i], C, N, nullptr);
      if (ret == KFE_OK)
      {
        passed++;
        if (verbose)
          fprintf(stderr, "PASSED\n");
      }
      else
      {
        failed++;
        if (verbose)
          fprintf(stderr, "FAILED\n");
      }
    }

    if (pool) {
      pm_free(pool, A);
      pm_free(pool, B);
      pm_free(pool, C);
      pm_free(pool, bias);
    } else {
      ffm_free(A);
      ffm_free(B);
      ffm_free(C);
      ffm_free(bias);
    }

    if (verbose)
    {
      fprintf(stderr, "\nSelf-test results: %d passed, %d failed\n", passed, failed);
    }

    return (failed == 0) ? KFE_OK : KFE_ERR_INTERNAL;
  }

  int kfe_benchmark_fusion(size_t m, size_t n, size_t k, int iterations)
  {
    // Placeholder - implemented in test file
    (void)m;
    (void)n;
    (void)k;
    (void)iterations;
    return KFE_OK;
  }

} // extern "C"