// // advanced/kFusion_engine/src/test_kernel_fusion_eve.cpp
//
// #include "kernel_fusion_eve.h"
// #include "kernel_fusion_engine.h"
// #include "mem_wrapper.h"
//
// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <cmath>
// #include <chrono>
//
// /* ==========================================================================
// */
// /* Helper Functions */
// /* ==========================================================================
// */
//
// static void init_matrix_random(float *M, size_t rows, size_t cols, size_t ld,
// float scale)
// {
//   for (size_t i = 0; i < rows; ++i)
//   {
//     for (size_t j = 0; j < cols; ++j)
//     {
//       M[i * ld + j] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * scale;
//     }
//   }
// }
//
// static void init_vector_constant(float *v, size_t n, float val)
// {
//   for (size_t i = 0; i < n; ++i)
//   {
//     v[i] = val;
//   }
// }
//
// static void init_vector_random(float *v, size_t n, float scale)
// {
//   for (size_t i = 0; i < n; ++i)
//   {
//     v[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * scale;
//   }
// }
//
// /* ==========================================================================
// /* Test Cases */
// /* ==========================================================================
// */
//
// static bool test_eve_basic()
// {
//   printf("TEST: EVE Basic Functionality\n");
//   printf("================================================================================\n");
//
//   printf("  EVE Available: %s\n", kfe_eve_is_available() ? "Yes" : "No");
//   printf("  EVE SIMD Width: %zu floats\n", kfe_eve_simd_width());
//   printf("  EVE Backend: %s\n", kfe_eve_backend_name());
//   printf("\n");
//
//   return kfe_eve_is_available();
// }
//
// static bool test_gemm_batchnorm_activation()
// {
//   printf("TEST: GEMM + BatchNorm + Activation (EVE-powered)\n");
//   printf("================================================================================\n");
//
//   const size_t M = 512, N = 256, K = 512;
//   const size_t n_channels = N;
//
//   float *A = static_cast<float *>(ffm_aligned_alloc(64, M * K * sizeof(float)));
//   float *B = static_cast<float *>(ffm_aligned_alloc(64, K * N * sizeof(float)));
//   float *bias = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *mean = static_cast<float *>(ffm_aligned_alloc(64, n_channels * sizeof(float)));
//   float *variance = static_cast<float *>(ffm_aligned_alloc(64, n_channels *sizeof(float)));
//   float *gamma = static_cast<float *>(ffm_aligned_alloc(64, n_channels * sizeof(float)));
//   float *beta = static_cast<float *>(ffm_aligned_alloc(64, n_channels *                                                                          sizeof(float)));
//   float *C = static_cast<float *>(ffm_aligned_alloc(64, M * N * sizeof(float)));
//
//   if (!A || !B || !bias || !mean || !variance || !gamma || !beta || !C)
//   {
//     printf("FAILED: Memory allocation\n\n");
//     return false;
//   }
//
//   init_matrix_random(A, M, K, K, 0.01f);
//   init_matrix_random(B, K, N, N, 0.01f);
//   init_vector_constant(bias, N, 0.001f);
//   init_vector_random(mean, n_channels, 0.1f);
//   init_vector_constant(variance, n_channels, 1.0f);
//   init_vector_constant(gamma, n_channels, 1.0f);
//   init_vector_constant(beta, n_channels, 0.0f);
//   memset(C, 0, M * N * sizeof(float));
//
//   kfe_batchnorm_params_t bn_params = {};
//   bn_params.mean = mean;
//   bn_params.variance = variance;
//   bn_params.gamma = gamma;
//   bn_params.beta = beta;
//   bn_params.epsilon = 1e-5f;
//
//   kfe_perf_stats_t stats = {};
//   int ret = kfe_sgemm_batchnorm_activation(
//       KFE_LAYOUT_ROW_MAJOR, KFE_NO_TRANS, KFE_NO_TRANS,
//       M, N, K, 1.0f, A, K, B, N, bias,
//       &bn_params, KFE_ACTIVATION_RELU,
//       C, N, &stats);
//
//   bool passed = (ret == KFE_OK);
//
//   // Sanity check outputs
//   bool has_invalid = false;
//   for (size_t i = 0; i < M * N && i < 1000; ++i)
//   {
//     if (std::isnan(C[i]) || std::isinf(C[i]))
//     {
//       has_invalid = true;
//       break;
//     }
//   }
//   passed = passed && !has_invalid;
//
//   printf("  Status: %s\n", passed ? "PASSED" : "FAILED");
//   printf("  Performance: %.2f GFLOPS\n", stats.gflops);
//   printf("  Elapsed: %.2f ms\n", stats.elapsed_ms);
//   printf("  Fused Ops: %zu (GEMM + Bias + BatchNorm + ReLU)\n",
//   stats.fused_ops_count); printf("  Memory Saved: %.2f MB\n",
//   stats.memory_saved_bytes / (1024.0 * 1024.0)); printf("  Pattern: %s\n",
//   stats.fusion_pattern); printf("\n");
//
//   ffm_free(A);
//   ffm_free(B);
//   ffm_free(bias);
//   ffm_free(mean);
//   ffm_free(variance);
//   ffm_free(gamma);
//   ffm_free(beta);
//   ffm_free(C);
//
//   return passed;
// }
//
// static bool test_gemm_layernorm_activation()
// {
//   printf("TEST: GEMM + LayerNorm + Activation (Transformer-style)\n");
//   printf("================================================================================\n");
//
//   const size_t M = 1024, N = 2048, K = 2048; // batch*seq_len × hidden_dim
//
//   float *A = static_cast<float *>(ffm_aligned_alloc(64, M * K * sizeof(float)));
//   float *B = static_cast<float *>(ffm_aligned_alloc(64, K * N * sizeof(float)));
//   float *bias = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *gamma = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *beta = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *C = static_cast<float *>(ffm_aligned_alloc(64, M * N * sizeof(float)));
//
//   if (!A || !B || !bias || !gamma || !beta || !C)
//   {
//     printf("FAILED: Memory allocation\n\n");
//     return false;
//   }
//
//   init_matrix_random(A, M, K, K, 0.01f);
//   init_matrix_random(B, K, N, N, 0.01f);
//   init_vector_constant(bias, N, 0.001f);
//   init_vector_constant(gamma, N, 1.0f);
//   init_vector_constant(beta, N, 0.0f);
//   memset(C, 0, M * N * sizeof(float));
//
//   kfe_layernorm_params_t ln_params = {};
//   ln_params.gamma = gamma;
//   ln_params.beta = beta;
//   ln_params.epsilon = 1e-5f;
//
//   kfe_perf_stats_t stats = {};
//   int ret = kfe_sgemm_layernorm_activation(
//       KFE_LAYOUT_ROW_MAJOR, KFE_NO_TRANS, KFE_NO_TRANS,
//       M, N, K, 1.0f, A, K, B, N, bias,
//       &ln_params, KFE_ACTIVATION_GELU,
//       C, N, &stats);
//
//   bool passed = (ret == KFE_OK);
//
//   // Sanity check
//   bool has_invalid = false;
//   for (size_t i = 0; i < M * N && i < 1000; ++i)
//   {
//     if (std::isnan(C[i]) || std::isinf(C[i]))
//     {
//       has_invalid = true;
//       break;
//     }
//   }
//   passed = passed && !has_invalid;
//
//   printf("  Status: %s\n", passed ? "PASSED" : "FAILED");
//   printf("  Performance: %.2f GFLOPS\n", stats.gflops);
//   printf("  Elapsed: %.2f ms\n", stats.elapsed_ms);
//   printf("  Fused Ops: %zu (GEMM + Bias + LayerNorm + GELU)\n",
//   stats.fused_ops_count); printf("  Memory Saved: %.2f MB\n",
//   stats.memory_saved_bytes / (1024.0 * 1024.0)); printf("  Pattern: %s\n",
//   stats.fusion_pattern); printf("\n");
//
//   ffm_free(A);
//   ffm_free(B);
//   ffm_free(bias);
//   ffm_free(gamma);
//   ffm_free(beta);
//   ffm_free(C);
//
//   return passed;
// }
//
// static bool test_eve_full_epilogue()
// {
//   printf("TEST: EVE Full Epilogue Fusion (Bias+BN+Residual+Activation)\n");
//   printf("================================================================================\n");
//
//   const size_t M = 256, N = 256, K = 256;
//
//   float *A = static_cast<float *>(ffm_aligned_alloc(64, M * K * sizeof(float)));
//   float *B = static_cast<float *>(ffm_aligned_alloc(64, K * N * sizeof(float)));
//   float *bias = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *mean = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *variance = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *gamma = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *beta_bn = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//   float *residual = static_cast<float *>(ffm_aligned_alloc(64, M * N * sizeof(float)));
//   float *C = static_cast<float *>(ffm_aligned_alloc(64, M * N * sizeof(float)));
//
//   if (!A || !B || !bias || !mean || !variance || !gamma || !beta_bn ||
//   !residual || !C)
//   {
//     printf("FAILED: Memory allocation\n\n");
//     return false;
//   }
//
//   init_matrix_random(A, M, K, K, 0.01f);
//   init_matrix_random(B, K, N, N, 0.01f);
//   init_vector_constant(bias, N, 0.001f);
//   init_vector_random(mean, N, 0.1f);
//   init_vector_constant(variance, N, 1.0f);
//   init_vector_constant(gamma, N, 1.0f);
//   init_vector_constant(beta_bn, N, 0.0f);
//   init_matrix_random(residual, M, N, N, 0.01f);
//   memset(C, 0, M * N * sizeof(float));
//
//   kfe_batchnorm_params_t bn_params = {};
//   bn_params.mean = mean;
//   bn_params.variance = variance;
//   bn_params.gamma = gamma;
//   bn_params.beta = beta_bn;
//   bn_params.epsilon = 1e-5f;
//
//   kfe_epilogue_config_t config = {};
//   config.enable_bias = 1;
//   config.enable_normalization = 1;
//   config.enable_activation = 1;
//   config.enable_residual = 1;
//   config.norm_type = KFE_NORM_BATCH;
//   config.activation = KFE_ACTIVATION_RELU;
//   config.use_eve_simd = 1;
//
//   kfe_perf_stats_t stats = {};
//   int ret = kfe_sgemm_eve_epilogue(
//       KFE_LAYOUT_ROW_MAJOR, KFE_NO_TRANS, KFE_NO_TRANS,
//       M, N, K, 1.0f, A, K, B, N, bias,
//       &bn_params, residual, N, &config,
//       C, N, &stats);
//
//   bool passed = (ret == KFE_OK);
//
//   // Sanity check
//   bool has_invalid = false;
//   for (size_t i = 0; i < M * N && i < 1000; ++i)
//   {
//     if (std::isnan(C[i]) || std::isinf(C[i]))
//     {
//       has_invalid = true;
//       break;
//     }
//   }
//   passed = passed && !has_invalid;
//
//   printf("  Status: %s\n", passed ? "PASSED" : "FAILED");
//   printf("  Performance: %.2f GFLOPS\n", stats.gflops);
//   printf("  Elapsed: %.2f ms\n", stats.elapsed_ms);
//   printf("  Fused Ops: %zu (GEMM + 4 epilogue ops)\n",
//   stats.fused_ops_count); printf("  Memory Saved: %.2f MB\n",
//   stats.memory_saved_bytes / (1024.0 * 1024.0)); printf("  Pattern: %s\n",
//   stats.fusion_pattern); printf("\n");
//
//   ffm_free(A);
//   ffm_free(B);
//   ffm_free(bias);
//   ffm_free(mean);
//   ffm_free(variance);
//   ffm_free(gamma);
//   ffm_free(beta_bn);
//   ffm_free(residual);
//   ffm_free(C);
//
//   return passed;
// }
//
// static void benchmark_eve_vs_scalar()
// {
//   printf("BENCHMARK: EVE SIMD vs Scalar Performance\n");
//   printf("================================================================================\n");
//
//   const size_t sizes[] = {512, 1024, 2048, 4096};
//   const int iterations = 20;
//
//   for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s)
//   {
//     size_t M = sizes[s], N = sizes[s], K = sizes[s];
//
//     float *A = static_cast<float *>(ffm_aligned_alloc(64, M * K * sizeof(float)));
//     float *B = static_cast<float *>(ffm_aligned_alloc(64, K * N * sizeof(float)));
//     float *bias = static_cast<float *>(ffm_aligned_alloc(64, N * sizeof(float)));
//     float *C = static_cast<float *>(ffm_aligned_alloc(64, M * N * sizeof(float)));
//
//     if (!A || !B || !bias || !C)
//       continue;
//
//     init_matrix_random(A, M, K, K, 0.01f);
//     init_matrix_random(B, K, N, N, 0.01f);
//     init_vector_constant(bias, N, 0.001f);
//
//     // Benchmark EVE version
//     auto t1 = std::chrono::high_resolution_clock::now();
//     for (int iter = 0; iter < iterations; ++iter)
//     {
//       kfe_sgemm_bias_activation(
//           KFE_LAYOUT_ROW_MAJOR, KFE_NO_TRANS, KFE_NO_TRANS,
//           M, N, K, 1.0f, A, K, B, N, bias,
//           KFE_ACTIVATION_GELU, C, N, nullptr);
//     }
//     auto t2 = std::chrono::high_resolution_clock::now();
//     double eve_time = std::chrono::duration<double, std::milli>(t2 -
//     t1).count();
//
//     printf("  Matrix Size: %zux%zux%zu\n", M, N, K);
//     printf("    EVE: %.2f ms/iter | %.2f GFLOPS\n",
//            eve_time / iterations,
//            (iterations * 2.0 * M * N * K) / (eve_time * 1e6));
//     printf("\n");
//
//     ffm_free(A);
//     ffm_free(B);
//     ffm_free(bias);
//     ffm_free(C);
//   }
// }
//
// /* ==========================================================================
// */
// /* Main Test Driver */
// /* ==========================================================================
// */
//
// int main(int argc, char **argv)
// {
//   printf("\n");
//   printf("################################################################################\n");
//   printf("# Kernel Fusion Engine - EVE SIMD Extension Test Suite #\n");
//   printf("################################################################################\n");
//   printf("\n");
//
//   srand(12345);
//
//   // Initialize KFE
//   kfe_config_t config = {};
//   config.num_threads = 0;
//   config.enable_vectorization = 1;
//   config.enable_cache_blocking = 1;
//   config.enable_prefetch = 1;
//   config.enable_kernel_autotuning = 1;
//   config.workspace_size_mb = 256;
//   config.verbose = 0;
//
//   int ret = kfe_init(&config);
//   if (ret != KFE_OK)
//   {
//     printf("FATAL: KFE initialization failed: %s\n", kfe_strerror(ret));
//     return 1;
//   }
//
//   printf("Kernel Fusion Engine initialized successfully\n");
//   printf("%s\n", kfe_get_system_info());
//   printf("\n");
//
//   // Run EVE tests
//   int passed = 0, failed = 0;
//
//   if (test_eve_basic())
//     passed++;
//   else
//     failed++;
//   if (test_gemm_batchnorm_activation())
//     passed++;
//   else
//     failed++;
//   if (test_gemm_layernorm_activation())
//     passed++;
//   else
//     failed++;
//   if (test_eve_full_epilogue())
//     passed++;
//   else
//     failed++;
//
//   // Run benchmarks
//   benchmark_eve_vs_scalar();
//
//   // Final summary
//   printf("################################################################################\n");
//   printf("# EVE Extension Test Summary #\n");
//   printf("################################################################################\n");
//   printf("\n");
//   printf("  Total Tests: %d\n", passed + failed);
//   printf("  Passed: %d\n", passed);
//   printf("  Failed: %d\n", failed);
//   printf("  Success Rate: %.1f%%\n", (100.0 * passed) / (passed + failed));
//   printf("\n");
//
//   kfe_shutdown();
//
//   return (failed == 0) ? 0 : 1;
// }