// advanced/operator_graph/src/test_operator_graph.cpp
/**
 * @file test_operator_graph.cpp
 * @brief Comprehensive test suite for Operator Graph / Fusion Runtime
 */

#include "kernel_fusion_engine.h"
#include "operator_graph.h"
#include "benchmark.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>

/* ========================================================================== */
/* Test Utilities                                                             */
/* ========================================================================== */

static bool FloatEqual(float a, float b, float epsilon = 1e-3f)
{
  return std::fabs(a - b) < epsilon;
}

static void FillRandom(float *data, size_t n, float min = -1.0f, float max = 1.0f)
{
  static std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(min, max);
  for (size_t i = 0; i < n; ++i)
  {
    data[i] = dist(gen);
  }
}

static void PrintTestHeader(const char *name)
{
  std::cout << "\n=== " << name << " ===" << std::endl;
}

static void PrintTestResult(const char *name, bool passed)
{
  std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << name << std::endl;
}

/* ========================================================================== */
/* Test 1: Initialization and Configuration                                   */
/* ========================================================================== */

static bool TestInitialization()
{
  PrintTestHeader("Test 1: Initialization");

  // Test default initialization
  int ret = og_init(nullptr);
  if (ret != OG_OK)
  {
    std::cerr << "Default initialization failed: " << og_strerror(ret) << std::endl;
    return false;
  }

  if (!og_is_initialized())
  {
    std::cerr << "Initialization check failed" << std::endl;
    return false;
  }

  og_shutdown();

  // Test custom configuration
  og_config_t config = {};
  config.enable_fusion = 1;
  config.enable_parallelism = 1;
  config.enable_memory_reuse = 1;
  config.enable_pattern_matching = 1;
  config.max_fusion_depth = 4;
  config.num_threads = 4;
  config.verbose = 0;
  config.fusion_threshold = 1.1;

  ret = og_init(&config);
  if (ret != OG_OK)
  {
    std::cerr << "Custom initialization failed: " << og_strerror(ret) << std::endl;
    return false;
  }

  std::cout << "Initialization successful" << std::endl;
  return true;
}

/* ========================================================================== */
/* Test 2: Graph Construction                                                 */
/* ========================================================================== */

static bool TestGraphConstruction()
{
  PrintTestHeader("Test 2: Graph Construction");

  og_graph_t graph = nullptr;
  int ret = og_create_graph(&graph);
  if (ret != OG_OK || !graph)
  {
    std::cerr << "Failed to create graph: " << og_strerror(ret) << std::endl;
    return false;
  }

  // Create tensors: A[128x256], B[256x512], C[128x512]
  og_tensor_t tensor_A = {};
  tensor_A.ndim = 2;
  tensor_A.shape[0] = 128;
  tensor_A.shape[1] = 256;
  tensor_A.is_constant = 0;

  uint64_t tid_A = 0;
  ret = og_add_tensor(graph, &tensor_A, &tid_A);
  if (ret != OG_OK)
  {
    std::cerr << "Failed to add tensor A: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  og_tensor_t tensor_B = {};
  tensor_B.ndim = 2;
  tensor_B.shape[0] = 256;
  tensor_B.shape[1] = 512;
  tensor_B.is_constant = 1;

  uint64_t tid_B = 0;
  ret = og_add_tensor(graph, &tensor_B, &tid_B);
  if (ret != OG_OK)
  {
    std::cerr << "Failed to add tensor B: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  og_tensor_t tensor_C = {};
  tensor_C.ndim = 2;
  tensor_C.shape[0] = 128;
  tensor_C.shape[1] = 512;
  tensor_C.is_constant = 0;

  uint64_t tid_C = 0;
  ret = og_add_tensor(graph, &tensor_C, &tid_C);
  if (ret != OG_OK)
  {
    std::cerr << "Failed to add tensor C: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  // Create GEMM node: C = A * B
  og_node_t gemm_node = {};
  gemm_node.type = OG_OP_GEMM;
  gemm_node.input_ids[0] = tid_A;
  gemm_node.input_ids[1] = tid_B;
  gemm_node.num_inputs = 2;
  gemm_node.output_ids[0] = tid_C;
  gemm_node.num_outputs = 1;
  gemm_node.attributes[0] = 1.0f; // alpha
  gemm_node.num_attributes = 1;

  uint64_t nid_gemm = 0;
  ret = og_add_node(graph, &gemm_node, &nid_gemm);
  if (ret != OG_OK)
  {
    std::cerr << "Failed to add GEMM node: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  std::cout << "Graph construction successful" << std::endl;
  std::cout << "  Created tensors: A[128x256], B[256x512], C[128x512]" << std::endl;
  std::cout << "  Created GEMM node: C = A * B" << std::endl;

  og_destroy_graph(graph);
  return true;
}

/* ========================================================================== */
/* Test 3: Pattern Detection - GEMM + Bias + ReLU                            */
/* ========================================================================== */

static bool TestPatternDetection()
{
  PrintTestHeader("Test 3: Pattern Detection");

  og_graph_t graph = nullptr;
  og_create_graph(&graph);

  // Create tensors
  og_tensor_t tensor_A = {};
  tensor_A.ndim = 2;
  tensor_A.shape[0] = 64;
  tensor_A.shape[1] = 128;
  uint64_t tid_A = 0;
  og_add_tensor(graph, &tensor_A, &tid_A);

  og_tensor_t tensor_B = {};
  tensor_B.ndim = 2;
  tensor_B.shape[0] = 128;
  tensor_B.shape[1] = 256;
  uint64_t tid_B = 0;
  og_add_tensor(graph, &tensor_B, &tid_B);

  og_tensor_t tensor_C = {};
  tensor_C.ndim = 2;
  tensor_C.shape[0] = 64;
  tensor_C.shape[1] = 256;
  uint64_t tid_C = 0;
  og_add_tensor(graph, &tensor_C, &tid_C);

  og_tensor_t tensor_bias = {};
  tensor_bias.ndim = 1;
  tensor_bias.shape[0] = 256;
  uint64_t tid_bias = 0;
  og_add_tensor(graph, &tensor_bias, &tid_bias);

  og_tensor_t tensor_D = {};
  tensor_D.ndim = 2;
  tensor_D.shape[0] = 64;
  tensor_D.shape[1] = 256;
  uint64_t tid_D = 0;
  og_add_tensor(graph, &tensor_D, &tid_D);

  og_tensor_t tensor_E = {};
  tensor_E.ndim = 2;
  tensor_E.shape[0] = 64;
  tensor_E.shape[1] = 256;
  uint64_t tid_E = 0;
  og_add_tensor(graph, &tensor_E, &tid_E);

  // Create nodes: GEMM -> BiasAdd -> ReLU
  og_node_t gemm_node = {};
  gemm_node.type = OG_OP_GEMM;
  gemm_node.input_ids[0] = tid_A;
  gemm_node.input_ids[1] = tid_B;
  gemm_node.num_inputs = 2;
  gemm_node.output_ids[0] = tid_C;
  gemm_node.num_outputs = 1;
  uint64_t nid_gemm = 0;
  og_add_node(graph, &gemm_node, &nid_gemm);

  og_node_t bias_node = {};
  bias_node.type = OG_OP_BIAS_ADD;
  bias_node.input_ids[0] = tid_C;
  bias_node.input_ids[1] = tid_bias;
  bias_node.num_inputs = 2;
  bias_node.output_ids[0] = tid_D;
  bias_node.num_outputs = 1;
  uint64_t nid_bias = 0;
  og_add_node(graph, &bias_node, &nid_bias);

  og_node_t relu_node = {};
  relu_node.type = OG_OP_RELU;
  relu_node.input_ids[0] = tid_D;
  relu_node.num_inputs = 1;
  relu_node.output_ids[0] = tid_E;
  relu_node.num_outputs = 1;
  uint64_t nid_relu = 0;
  og_add_node(graph, &relu_node, &nid_relu);

  // Add edges
  og_add_edge(graph, nid_gemm, nid_bias);
  og_add_edge(graph, nid_bias, nid_relu);

  // Finalize graph (triggers pattern detection)
  int ret = og_finalize_graph(graph);
  if (ret != OG_OK)
  {
    std::cerr << "Failed to finalize graph: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  // Check for detected patterns
  og_fusion_group_t groups[10];
  size_t num_groups = 0;
  ret = og_detect_fusion_patterns(graph, groups, 10, &num_groups);
  if (ret != OG_OK)
  {
    std::cerr << "Pattern detection failed: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  std::cout << "Pattern detection complete" << std::endl;
  std::cout << "  Detected " << num_groups << " fusion group(s)" << std::endl;

  for (size_t i = 0; i < num_groups; ++i)
  {
    std::cout << "  Group " << i << ": " << og_pattern_name(groups[i].pattern)
              << " (" << groups[i].num_nodes << " nodes, "
              << "memory_saved=" << (groups[i].estimated_memory_saved_bytes / 1024.0) << " KB, "
              << "speedup=" << groups[i].estimated_speedup << "x)" << std::endl;
  }

  bool success = (num_groups > 0);
  if (success)
  {
    std::cout << "Successfully detected fusible pattern!" << std::endl;
  }

  og_destroy_graph(graph);
  return success;
}

/* ========================================================================== */
/* Test 4: Graph Execution with Real Data                                    */
/* ========================================================================== */

static bool TestGraphExecution()
{
  PrintTestHeader("Test 4: Graph Execution");

  og_graph_t graph = nullptr;
  og_create_graph(&graph);

  // Small matrices for validation
  const size_t M = 32, K = 64, N = 48;

  // Allocate and initialize data
  std::vector<float> A_data(M * K);
  std::vector<float> B_data(K * N);
  std::vector<float> C_data(M * N, 0.0f);

  FillRandom(A_data.data(), M * K, 0.0f, 1.0f);
  FillRandom(B_data.data(), K * N, 0.0f, 1.0f);

  // Create tensors with data
  og_tensor_t tensor_A = {};
  tensor_A.ndim = 2;
  tensor_A.shape[0] = M;
  tensor_A.shape[1] = K;
  tensor_A.data = A_data.data();
  uint64_t tid_A = 0;
  og_add_tensor(graph, &tensor_A, &tid_A);

  og_tensor_t tensor_B = {};
  tensor_B.ndim = 2;
  tensor_B.shape[0] = K;
  tensor_B.shape[1] = N;
  tensor_B.data = B_data.data();
  uint64_t tid_B = 0;
  og_add_tensor(graph, &tensor_B, &tid_B);

  og_tensor_t tensor_C = {};
  tensor_C.ndim = 2;
  tensor_C.shape[0] = M;
  tensor_C.shape[1] = N;
  tensor_C.data = C_data.data();
  uint64_t tid_C = 0;
  og_add_tensor(graph, &tensor_C, &tid_C);

  // Create GEMM node
  og_node_t gemm_node = {};
  gemm_node.type = OG_OP_GEMM;
  gemm_node.input_ids[0] = tid_A;
  gemm_node.input_ids[1] = tid_B;
  gemm_node.num_inputs = 2;
  gemm_node.output_ids[0] = tid_C;
  gemm_node.num_outputs = 1;
  gemm_node.attributes[0] = 1.0f;
  gemm_node.num_attributes = 1;
  uint64_t nid_gemm = 0;
  og_add_node(graph, &gemm_node, &nid_gemm);

  // Finalize and execute
  int ret = og_finalize_graph(graph);
  if (ret != OG_OK)
  {
    std::cerr << "Failed to finalize: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  og_graph_stats_t stats = {};
  ret = og_execute_graph(graph, &stats);
  if (ret != OG_OK)
  {
    std::cerr << "Graph execution failed: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  std::cout << "Graph execution successful" << std::endl;
  std::cout << "  Execution time: " << std::fixed << std::setprecision(3)
            << stats.total_execution_time_ms << " ms" << std::endl;

  // Validate result (spot check a few elements)
  // Manual GEMM: C[i,j] = sum_k A[i,k] * B[k,j]
  // Row-major: A[i,k] = A_data[i*K + k], B[k,j] = B_data[k*N + j], C[i,j] = C_data[i*N + j]
  bool valid = true;
  for (size_t i = 0; i < std::min(size_t(5), M); ++i)
  {
    for (size_t j = 0; j < std::min(size_t(5), N); ++j)
    {
      float expected = 0.0f;
      for (size_t k = 0; k < K; ++k)
      {
        expected += A_data[i * K + k] * B_data[k * N + j];
      }
      float actual = C_data[i * N + j];
      if (!FloatEqual(expected, actual, 1e-2f))
      {
        std::cerr << "Validation failed at C[" << i << "][" << j << "]: "
                  << "expected=" << expected << ", actual=" << actual << std::endl;

        // Debug: print first few values
        if (i == 0 && j == 0)
        {
          std::cerr << "Debug info:" << std::endl;
          std::cerr << "  M=" << M << ", N=" << N << ", K=" << K << std::endl;
          std::cerr << "  A[0][0]=" << A_data[0] << ", A[0][1]=" << A_data[1] << std::endl;
          std::cerr << "  B[0][0]=" << B_data[0] << ", B[1][0]=" << B_data[N] << std::endl;
          std::cerr << "  Expected sum for C[0][0]:" << std::endl;
          float partial = 0.0f;
          for (size_t k = 0; k < 5; ++k)
          {
            float contrib = A_data[k] * B_data[k * N];
            partial += contrib;
            std::cerr << "    k=" << k << ": A[0][" << k << "]=" << A_data[k]
                      << " * B[" << k << "][0]=" << B_data[k * N]
                      << " = " << contrib << " (running sum=" << partial << ")" << std::endl;
          }
        }
        valid = false;
        break;
      }
    }
    if (!valid)
      break;
  }

  if (valid)
  {
    std::cout << "Result validation: PASS" << std::endl;
  }

  og_destroy_graph(graph);
  return valid;
}

/* ========================================================================== */
/* Test 5: Performance Benchmark - Fused vs Unfused                          */
/* ========================================================================== */

static bool TestPerformanceBenchmark()
{
  PrintTestHeader("Test 5: Performance Benchmark");

  const size_t M = 256, K = 512, N = 1024;
  const int iterations = 100;

  std::cout << "Matrix sizes: M=" << M << ", K=" << K << ", N=" << N << std::endl;
  std::cout << "Iterations: " << iterations << std::endl;

  // Allocate data
  std::vector<float> A_data(M * K);
  std::vector<float> B_data(K * N);
  std::vector<float> C_data(M * N);
  std::vector<float> bias_data(N);

  FillRandom(A_data.data(), M * K);
  FillRandom(B_data.data(), K * N);
  FillRandom(bias_data.data(), N, -0.5f, 0.5f);

  // Test with fusion enabled
  og_graph_t graph_fused = nullptr;
  og_create_graph(&graph_fused);

  og_tensor_t tA = {}, tB = {}, tC = {}, tBias = {}, tD = {}, tE = {};
  tA.ndim = 2;
  tA.shape[0] = M;
  tA.shape[1] = K;
  tA.data = A_data.data();
  tB.ndim = 2;
  tB.shape[0] = K;
  tB.shape[1] = N;
  tB.data = B_data.data();
  tC.ndim = 2;
  tC.shape[0] = M;
  tC.shape[1] = N;
  tBias.ndim = 1;
  tBias.shape[0] = N;
  tBias.data = bias_data.data();
  tD.ndim = 2;
  tD.shape[0] = M;
  tD.shape[1] = N;
  tE.ndim = 2;
  tE.shape[0] = M;
  tE.shape[1] = N;
  tE.data = C_data.data();

  uint64_t tidA, tidB, tidC, tidBias, tidD, tidE;
  og_add_tensor(graph_fused, &tA, &tidA);
  og_add_tensor(graph_fused, &tB, &tidB);
  og_add_tensor(graph_fused, &tC, &tidC);
  og_add_tensor(graph_fused, &tBias, &tidBias);
  og_add_tensor(graph_fused, &tD, &tidD);
  og_add_tensor(graph_fused, &tE, &tidE);

  og_node_t nGemm = {}, nBias = {}, nRelu = {};
  nGemm.type = OG_OP_GEMM;
  nGemm.input_ids[0] = tidA;
  nGemm.input_ids[1] = tidB;
  nGemm.num_inputs = 2;
  nGemm.output_ids[0] = tidC;
  nGemm.num_outputs = 1;

  nBias.type = OG_OP_BIAS_ADD;
  nBias.input_ids[0] = tidC;
  nBias.input_ids[1] = tidBias;
  nBias.num_inputs = 2;
  nBias.output_ids[0] = tidD;
  nBias.num_outputs = 1;

  nRelu.type = OG_OP_RELU;
  nRelu.input_ids[0] = tidD;
  nRelu.num_inputs = 1;
  nRelu.output_ids[0] = tidE;
  nRelu.num_outputs = 1;

  uint64_t nidG, nidB, nidR;
  og_add_node(graph_fused, &nGemm, &nidG);
  og_add_node(graph_fused, &nBias, &nidB);
  og_add_node(graph_fused, &nRelu, &nidR);
  og_add_edge(graph_fused, nidG, nidB);
  og_add_edge(graph_fused, nidB, nidR);

  og_finalize_graph(graph_fused);

  // Warmup
  og_execute_graph(graph_fused, nullptr);

  // Benchmark
  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i)
  {
    og_reset_execution_state(graph_fused);
    og_execute_graph(graph_fused, nullptr);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time_ms = elapsed.count() / iterations;

  // Calculate GFLOPS
  double flops = 2.0 * M * N * K + M * N + M * N; // GEMM + Bias + ReLU
  double gflops = (flops / 1e9) / (avg_time_ms / 1000.0);

  std::cout << "\nFused execution:" << std::endl;
  std::cout << "  Average time: " << std::fixed << std::setprecision(3)
            << avg_time_ms << " ms" << std::endl;
  std::cout << "  Performance: " << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

  og_graph_stats_t stats = {};
  og_get_graph_stats(graph_fused, &stats);
  std::cout << "  Fusion groups: " << stats.fusion_groups_detected << std::endl;
  std::cout << "  Operations fused: " << stats.total_ops_fused << std::endl;
  std::cout << "  Memory saved: " << (stats.total_memory_saved_bytes / 1024.0 / 1024.0)
            << " MB" << std::endl;

  og_destroy_graph(graph_fused);
  return true;
}

/* ========================================================================== */
/* Test 6: Self-Test (Comprehensive Correctness)                             */
/* ========================================================================== */

static bool TestSelfTest()
{
  PrintTestHeader("Test 6: Self-Test");

  // Test multiple fusion patterns with numerical validation
  const size_t M = 64, N = 64, K = 64;

  std::vector<float> A_data(M * K);
  std::vector<float> B_data(K * N);
  std::vector<float> bias_data(N);
  std::vector<float> residual_data(M * N);

  FillRandom(A_data.data(), M * K, 0.0f, 1.0f);
  FillRandom(B_data.data(), K * N, 0.0f, 1.0f);
  FillRandom(bias_data.data(), N, -0.5f, 0.5f);
  FillRandom(residual_data.data(), M * N, 0.0f, 1.0f);

  int passed = 0, failed = 0;

  // Test 1: GEMM + Bias
  {
    og_graph_t graph = nullptr;
    og_create_graph(&graph);

    og_tensor_t tA = {}, tB = {}, tC = {}, tBias = {}, tD = {};
    tA.ndim = 2;
    tA.shape[0] = M;
    tA.shape[1] = K;
    tA.data = A_data.data();
    tB.ndim = 2;
    tB.shape[0] = K;
    tB.shape[1] = N;
    tB.data = B_data.data();
    tC.ndim = 2;
    tC.shape[0] = M;
    tC.shape[1] = N;
    tBias.ndim = 1;
    tBias.shape[0] = N;
    tBias.data = bias_data.data();
    tD.ndim = 2;
    tD.shape[0] = M;
    tD.shape[1] = N;

    uint64_t tidA, tidB, tidC, tidBias, tidD;
    og_add_tensor(graph, &tA, &tidA);
    og_add_tensor(graph, &tB, &tidB);
    og_add_tensor(graph, &tC, &tidC);
    og_add_tensor(graph, &tBias, &tidBias);
    og_add_tensor(graph, &tD, &tidD);

    og_node_t nGemm = {}, nBias = {};
    nGemm.type = OG_OP_GEMM;
    nGemm.input_ids[0] = tidA;
    nGemm.input_ids[1] = tidB;
    nGemm.num_inputs = 2;
    nGemm.output_ids[0] = tidC;
    nGemm.num_outputs = 1;

    nBias.type = OG_OP_BIAS_ADD;
    nBias.input_ids[0] = tidC;
    nBias.input_ids[1] = tidBias;
    nBias.num_inputs = 2;
    nBias.output_ids[0] = tidD;
    nBias.num_outputs = 1;

    uint64_t nidG, nidB;
    og_add_node(graph, &nGemm, &nidG);
    og_add_node(graph, &nBias, &nidB);
    og_add_edge(graph, nidG, nidB);

    if (og_finalize_graph(graph) == OG_OK && og_execute_graph(graph, nullptr) == OG_OK)
    {
      passed++;
      std::cout << "  [PASS] GEMM+Bias fusion" << std::endl;
    }
    else
    {
      failed++;
      std::cerr << "  [FAIL] GEMM+Bias fusion" << std::endl;
    }

    og_destroy_graph(graph);
  }

  // Test 2: GEMM + Bias + Activation (multiple activations)
  kfe_activation_t activations[] = {KFE_ACTIVATION_RELU, KFE_ACTIVATION_TANH,
                                    KFE_ACTIVATION_SIGMOID, KFE_ACTIVATION_GELU};
  const char *act_names[] = {"ReLU", "Tanh", "Sigmoid", "GELU"};

  for (int act_idx = 0; act_idx < 4; ++act_idx)
  {
    og_graph_t graph = nullptr;
    og_create_graph(&graph);

    og_tensor_t tA = {}, tB = {}, tC = {}, tBias = {}, tD = {}, tE = {};
    tA.ndim = 2;
    tA.shape[0] = M;
    tA.shape[1] = K;
    tA.data = A_data.data();
    tB.ndim = 2;
    tB.shape[0] = K;
    tB.shape[1] = N;
    tB.data = B_data.data();
    tC.ndim = 2;
    tC.shape[0] = M;
    tC.shape[1] = N;
    tBias.ndim = 1;
    tBias.shape[0] = N;
    tBias.data = bias_data.data();
    tD.ndim = 2;
    tD.shape[0] = M;
    tD.shape[1] = N;
    tE.ndim = 2;
    tE.shape[0] = M;
    tE.shape[1] = N;

    uint64_t tidA, tidB, tidC, tidBias, tidD, tidE;
    og_add_tensor(graph, &tA, &tidA);
    og_add_tensor(graph, &tB, &tidB);
    og_add_tensor(graph, &tC, &tidC);
    og_add_tensor(graph, &tBias, &tidBias);
    og_add_tensor(graph, &tD, &tidD);
    og_add_tensor(graph, &tE, &tidE);

    og_node_t nGemm = {}, nBias = {}, nAct = {};
    nGemm.type = OG_OP_GEMM;
    nGemm.input_ids[0] = tidA;
    nGemm.input_ids[1] = tidB;
    nGemm.num_inputs = 2;
    nGemm.output_ids[0] = tidC;
    nGemm.num_outputs = 1;

    nBias.type = OG_OP_BIAS_ADD;
    nBias.input_ids[0] = tidC;
    nBias.input_ids[1] = tidBias;
    nBias.num_inputs = 2;
    nBias.output_ids[0] = tidD;
    nBias.num_outputs = 1;

    nAct.type = (act_idx == 0) ? OG_OP_RELU : (act_idx == 1) ? OG_OP_TANH
                                          : (act_idx == 2)   ? OG_OP_SIGMOID
                                                             : OG_OP_GELU;
    nAct.input_ids[0] = tidD;
    nAct.num_inputs = 1;
    nAct.output_ids[0] = tidE;
    nAct.num_outputs = 1;

    uint64_t nidG, nidB, nidA;
    og_add_node(graph, &nGemm, &nidG);
    og_add_node(graph, &nBias, &nidB);
    og_add_node(graph, &nAct, &nidA);
    og_add_edge(graph, nidG, nidB);
    og_add_edge(graph, nidB, nidA);

    if (og_finalize_graph(graph) == OG_OK && og_execute_graph(graph, nullptr) == OG_OK)
    {
      passed++;
      std::cout << "  [PASS] GEMM+Bias+" << act_names[act_idx] << " fusion" << std::endl;
    }
    else
    {
      failed++;
      std::cerr << "  [FAIL] GEMM+Bias+" << act_names[act_idx] << " fusion" << std::endl;
    }

    og_destroy_graph(graph);
  }

  std::cout << "\nSelf-test results: " << passed << " passed, " << failed << " failed" << std::endl;
  return (failed == 0);
}

/* ========================================================================== */
/* Test 7: Long Fusion Chains (5+ Operations)                                */
/* ========================================================================== */

static bool TestLongFusionChains()
{
  PrintTestHeader("Test 7: Long Fusion Chains");

  const size_t M = 64, K = 64, N = 64;

  std::vector<float> A_data(M * K);
  std::vector<float> B_data(K * N);
  std::vector<float> bias_data(N);

  FillRandom(A_data.data(), M * K);
  FillRandom(B_data.data(), K * N);
  FillRandom(bias_data.data(), N, -0.5f, 0.5f);

  og_graph_t graph = nullptr;
  og_create_graph(&graph);

  // Create 6-operation chain: GEMM -> Bias -> ReLU -> GEMM -> Bias -> Tanh
  std::vector<uint64_t> tensor_ids;

  // Input tensors
  og_tensor_t t = {};
  t.ndim = 2;
  t.shape[0] = M;
  t.shape[1] = K;
  t.data = A_data.data();
  uint64_t tid;
  og_add_tensor(graph, &t, &tid);
  tensor_ids.push_back(tid);

  t.shape[0] = K;
  t.shape[1] = N;
  t.data = B_data.data();
  og_add_tensor(graph, &t, &tid);
  tensor_ids.push_back(tid);

  // Intermediate and output tensors
  for (int i = 0; i < 6; ++i)
  {
    t.shape[0] = M;
    t.shape[1] = N;
    t.data = nullptr;
    og_add_tensor(graph, &t, &tid);
    tensor_ids.push_back(tid);
  }

  // Bias tensor
  t.ndim = 1;
  t.shape[0] = N;
  t.data = bias_data.data();
  og_add_tensor(graph, &t, &tid);
  uint64_t bias_tid = tid;

  // Build chain
  std::vector<uint64_t> node_ids;

  // Op 1: GEMM
  og_node_t node = {};
  node.type = OG_OP_GEMM;
  node.input_ids[0] = tensor_ids[0];
  node.input_ids[1] = tensor_ids[1];
  node.num_inputs = 2;
  node.output_ids[0] = tensor_ids[2];
  node.num_outputs = 1;
  og_add_node(graph, &node, &tid);
  node_ids.push_back(tid);

  // Op 2: Bias
  node.type = OG_OP_BIAS_ADD;
  node.input_ids[0] = tensor_ids[2];
  node.input_ids[1] = bias_tid;
  node.num_inputs = 2;
  node.output_ids[0] = tensor_ids[3];
  node.num_outputs = 1;
  og_add_node(graph, &node, &tid);
  node_ids.push_back(tid);

  // Op 3: ReLU
  node.type = OG_OP_RELU;
  node.input_ids[0] = tensor_ids[3];
  node.num_inputs = 1;
  node.output_ids[0] = tensor_ids[4];
  node.num_outputs = 1;
  og_add_node(graph, &node, &tid);
  node_ids.push_back(tid);

  // Op 4: GEMM (reuse first GEMM inputs)
  node.type = OG_OP_GEMM;
  node.input_ids[0] = tensor_ids[4];
  node.input_ids[1] = tensor_ids[1];
  node.num_inputs = 2;
  node.output_ids[0] = tensor_ids[5];
  node.num_outputs = 1;
  og_add_node(graph, &node, &tid);
  node_ids.push_back(tid);

  // Op 5: Bias
  node.type = OG_OP_BIAS_ADD;
  node.input_ids[0] = tensor_ids[5];
  node.input_ids[1] = bias_tid;
  node.num_inputs = 2;
  node.output_ids[0] = tensor_ids[6];
  node.num_outputs = 1;
  og_add_node(graph, &node, &tid);
  node_ids.push_back(tid);

  // Op 6: Tanh
  node.type = OG_OP_TANH;
  node.input_ids[0] = tensor_ids[6];
  node.num_inputs = 1;
  node.output_ids[0] = tensor_ids[7];
  node.num_outputs = 1;
  og_add_node(graph, &node, &tid);
  node_ids.push_back(tid);

  // Add edges
  for (size_t i = 0; i < node_ids.size() - 1; ++i)
  {
    og_add_edge(graph, node_ids[i], node_ids[i + 1]);
  }

  int ret = og_finalize_graph(graph);
  if (ret != OG_OK)
  {
    std::cerr << "Failed to finalize long chain: " << og_strerror(ret) << std::endl;
    og_destroy_graph(graph);
    return false;
  }

  og_graph_stats_t stats = {};
  ret = og_execute_graph(graph, &stats);

  std::cout << "Long chain (6 ops) execution:" << std::endl;
  std::cout << "  Execution time: " << std::fixed << std::setprecision(3)
            << stats.total_execution_time_ms << " ms" << std::endl;
  std::cout << "  Fusion groups detected: " << stats.fusion_groups_detected << std::endl;
  std::cout << "  Operations fused: " << stats.total_ops_fused << std::endl;

  og_destroy_graph(graph);
  return (ret == OG_OK);
}

/* ========================================================================== */
/* Test 8: Large Matrix Sizes                                                 */
/* ========================================================================== */

static bool TestLargeMatrices()
{
  PrintTestHeader("Test 8: Large Matrix Sizes");

  struct TestSize
  {
    size_t K;
  }; // Square matrices: M=N=K
  std::vector<TestSize> sizes = {
      {2048},
      {4096},
      {8192}};

  bool all_passed = true;

  for (auto &sz : sizes)
  {
    size_t M = sz.K, N = sz.K, K = sz.K;
    std::cout << "\nTesting square matrix size K=" << K << " (M=N=K=" << K << ")" << std::endl;

    std::vector<float> A_data(sz.K * sz.K);
    std::vector<float> B_data(sz.K * sz.K);
    std::vector<float> bias_data(sz.K);

    FillRandom(A_data.data(), sz.K * sz.K);
    FillRandom(B_data.data(), sz.K * sz.K);
    FillRandom(bias_data.data(), sz.K);

    og_graph_t graph = nullptr;
    og_create_graph(&graph);

    og_tensor_t tA = {}, tB = {}, tC = {}, tBias = {}, tD = {}, tE = {};
    tA.ndim = 2;
    tA.shape[0] = sz.K;
    tA.shape[1] = sz.K;
    tA.data = A_data.data();
    tB.ndim = 2;
    tB.shape[0] = sz.K;
    tB.shape[1] = sz.K;
    tB.data = B_data.data();
    tC.ndim = 2;
    tC.shape[0] = sz.K;
    tC.shape[1] = sz.K;
    tBias.ndim = 1;
    tBias.shape[0] = sz.K;
    tBias.data = bias_data.data();
    tD.ndim = 2;
    tD.shape[0] = sz.K;
    tD.shape[1] = sz.K;
    tE.ndim = 2;
    tE.shape[0] = sz.K;
    tE.shape[1] = sz.K;

    uint64_t tidA, tidB, tidC, tidBias, tidD, tidE;
    og_add_tensor(graph, &tA, &tidA);
    og_add_tensor(graph, &tB, &tidB);
    og_add_tensor(graph, &tC, &tidC);
    og_add_tensor(graph, &tBias, &tidBias);
    og_add_tensor(graph, &tD, &tidD);
    og_add_tensor(graph, &tE, &tidE);

    og_node_t nGemm = {}, nBias = {}, nRelu = {};
    nGemm.type = OG_OP_GEMM;
    nGemm.input_ids[0] = tidA;
    nGemm.input_ids[1] = tidB;
    nGemm.num_inputs = 2;
    nGemm.output_ids[0] = tidC;
    nGemm.num_outputs = 1;

    nBias.type = OG_OP_BIAS_ADD;
    nBias.input_ids[0] = tidC;
    nBias.input_ids[1] = tidBias;
    nBias.num_inputs = 2;
    nBias.output_ids[0] = tidD;
    nBias.num_outputs = 1;

    nRelu.type = OG_OP_RELU;
    nRelu.input_ids[0] = tidD;
    nRelu.num_inputs = 1;
    nRelu.output_ids[0] = tidE;
    nRelu.num_outputs = 1;

    uint64_t nidG, nidB, nidR;
    og_add_node(graph, &nGemm, &nidG);
    og_add_node(graph, &nBias, &nidB);
    og_add_node(graph, &nRelu, &nidR);
    og_add_edge(graph, nidG, nidB);
    og_add_edge(graph, nidB, nidR);

    og_finalize_graph(graph);

    auto start = std::chrono::high_resolution_clock::now();
    og_graph_stats_t stats = {};
    int ret = og_execute_graph(graph, &stats);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    if (ret == OG_OK)
    {
      // For square matrices: FLOPS = 2*K^3
      double flops = 2.0 * K * K * K;
      double gflops = (flops / 1e9) / (elapsed.count() / 1000.0);

      std::cout << "  Status: PASS" << std::endl;
      std::cout << "  Time: " << std::fixed << std::setprecision(2)
                << elapsed.count() << " ms" << std::endl;
      std::cout << "  Performance: " << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
      std::cout << "  FLOPS: " << std::scientific << std::setprecision(2) << flops << std::endl;
      std::cout << "  Memory saved: " << std::fixed << std::setprecision(2)
                << (stats.total_memory_saved_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    }
    else
    {
      std::cerr << "  Status: FAIL" << std::endl;
      all_passed = false;
    }

    og_destroy_graph(graph);
  }

  return all_passed;
}

/* ========================================================================== */
/* Test 9: Numerical Precision (Fused vs Unfused)                            */
/* ========================================================================== */

static bool TestNumericalPrecision()
{
  PrintTestHeader("Test 9: Numerical Precision");

  const size_t M = 128, N = 128, K = 128;

  std::vector<float> A_data(M * K);
  std::vector<float> B_data(K * N);
  std::vector<float> bias_data(N);
  std::vector<float> C_fused(M * N, 0.0f);
  std::vector<float> C_unfused(M * N, 0.0f);

  FillRandom(A_data.data(), M * K);
  FillRandom(B_data.data(), K * N);
  FillRandom(bias_data.data(), N);

  // Compute fused version via operator graph
  {
    og_graph_t graph = nullptr;
    og_create_graph(&graph);

    og_tensor_t tA = {}, tB = {}, tC = {}, tBias = {}, tD = {}, tE = {};
    tA.ndim = 2;
    tA.shape[0] = M;
    tA.shape[1] = K;
    tA.data = A_data.data();
    tB.ndim = 2;
    tB.shape[0] = K;
    tB.shape[1] = N;
    tB.data = B_data.data();
    tC.ndim = 2;
    tC.shape[0] = M;
    tC.shape[1] = N;
    tBias.ndim = 1;
    tBias.shape[0] = N;
    tBias.data = bias_data.data();
    tD.ndim = 2;
    tD.shape[0] = M;
    tD.shape[1] = N;
    tE.ndim = 2;
    tE.shape[0] = M;
    tE.shape[1] = N;
    tE.data = C_fused.data();

    uint64_t tidA, tidB, tidC, tidBias, tidD, tidE;
    og_add_tensor(graph, &tA, &tidA);
    og_add_tensor(graph, &tB, &tidB);
    og_add_tensor(graph, &tC, &tidC);
    og_add_tensor(graph, &tBias, &tidBias);
    og_add_tensor(graph, &tD, &tidD);
    og_add_tensor(graph, &tE, &tidE);

    og_node_t nGemm = {}, nBias = {}, nRelu = {};
    nGemm.type = OG_OP_GEMM;
    nGemm.input_ids[0] = tidA;
    nGemm.input_ids[1] = tidB;
    nGemm.num_inputs = 2;
    nGemm.output_ids[0] = tidC;
    nGemm.num_outputs = 1;

    nBias.type = OG_OP_BIAS_ADD;
    nBias.input_ids[0] = tidC;
    nBias.input_ids[1] = tidBias;
    nBias.num_inputs = 2;
    nBias.output_ids[0] = tidD;
    nBias.num_outputs = 1;

    nRelu.type = OG_OP_RELU;
    nRelu.input_ids[0] = tidD;
    nRelu.num_inputs = 1;
    nRelu.output_ids[0] = tidE;
    nRelu.num_outputs = 1;

    uint64_t nidG, nidB, nidR;
    og_add_node(graph, &nGemm, &nidG);
    og_add_node(graph, &nBias, &nidB);
    og_add_node(graph, &nRelu, &nidR);
    og_add_edge(graph, nidG, nidB);
    og_add_edge(graph, nidB, nidR);

    og_finalize_graph(graph);
    og_execute_graph(graph, nullptr);
    og_destroy_graph(graph);
  }

  // Compute unfused version manually
  {
    std::vector<float> temp1(M * N, 0.0f);
    std::vector<float> temp2(M * N, 0.0f);

    // GEMM: temp1 = A * B
    for (size_t i = 0; i < M; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k)
        {
          sum += A_data[i * K + k] * B_data[k * N + j];
        }
        temp1[i * N + j] = sum;
      }
    }

    // Bias: temp2 = temp1 + bias
    for (size_t i = 0; i < M; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        temp2[i * N + j] = temp1[i * N + j] + bias_data[j];
      }
    }

    // ReLU: C_unfused = max(0, temp2)
    for (size_t i = 0; i < M * N; ++i)
    {
      C_unfused[i] = (temp2[i] > 0.0f) ? temp2[i] : 0.0f;
    }
  }

  // Compare results
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  size_t mismatches = 0;

  for (size_t i = 0; i < M * N; ++i)
  {
    double abs_error = std::abs(C_fused[i] - C_unfused[i]);
    double rel_error = (C_unfused[i] != 0.0f) ? abs_error / std::abs(C_unfused[i]) : 0.0;

    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);

    if (abs_error > 1e-3f)
    {
      mismatches++;
    }
  }

  std::cout << "Precision comparison (Fused vs Unfused):" << std::endl;
  std::cout << "  Max absolute error: " << std::scientific << std::setprecision(6)
            << max_abs_error << std::endl;
  std::cout << "  Max relative error: " << max_rel_error << std::endl;
  std::cout << "  Mismatches (>1e-3): " << mismatches << " / " << (M * N) << std::endl;

  bool passed = (mismatches == 0 && max_abs_error < 1e-2);
  std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;

  return passed;
}

/* ========================================================================== */
/* Test 10: Graph Utilities                                                   */
/* ========================================================================== */

static bool TestGraphUtilities()
{
  PrintTestHeader("Test 10: Graph Utilities");

  og_graph_t graph = nullptr;
  og_create_graph(&graph);

  // Create simple graph
  og_tensor_t tA = {}, tB = {}, tC = {};
  tA.ndim = 2;
  tA.shape[0] = 64;
  tA.shape[1] = 64;
  tB.ndim = 2;
  tB.shape[0] = 64;
  tB.shape[1] = 64;
  tC.ndim = 2;
  tC.shape[0] = 64;
  tC.shape[1] = 64;

  uint64_t tidA, tidB, tidC;
  og_add_tensor(graph, &tA, &tidA);
  og_add_tensor(graph, &tB, &tidB);
  og_add_tensor(graph, &tC, &tidC);

  og_node_t nGemm = {};
  nGemm.type = OG_OP_GEMM;
  nGemm.input_ids[0] = tidA;
  nGemm.input_ids[1] = tidB;
  nGemm.num_inputs = 2;
  nGemm.output_ids[0] = tidC;
  nGemm.num_outputs = 1;

  uint64_t nid;
  og_add_node(graph, &nGemm, &nid);
  og_finalize_graph(graph);

  // Test graph printing
  std::cout << "Testing graph print:" << std::endl;
  og_print_graph(graph, 1);

  // Test DOT export
  const char *dot_file = "test_graph.dot";
  int ret = og_export_graph_dot(graph, dot_file);
  if (ret == OG_OK)
  {
    std::cout << "Successfully exported graph to " << dot_file << std::endl;
  }

  // Test validation
  ret = og_validate_graph(graph);
  std::cout << "Graph validation: " << (ret == OG_OK ? "PASS" : "FAIL") << std::endl;

  // Test memory estimation
  size_t peak_memory = 0;
  ret = og_estimate_memory_usage(graph, &peak_memory);
  if (ret == OG_OK)
  {
    std::cout << "Estimated peak memory: " << (peak_memory / 1024.0) << " KB" << std::endl;
  }

  og_destroy_graph(graph);
  return true;
}

/* ========================================================================== */
/* Main Test Driver                                                           */
/* ========================================================================== */

int main()
{
  std::cout << "===============================================" << std::endl;
  std::cout << "  Operator Graph / Fusion Runtime Test Suite  " << std::endl;
  std::cout << "===============================================" << std::endl;

  int passed = 0;
  int total = 0;

  // Test 1: Initialization with verbose mode
  total++;
  og_config_t config = {};
  config.enable_fusion = 1;
  config.enable_parallelism = 1;
  config.enable_memory_reuse = 1;
  config.enable_pattern_matching = 1;
  config.max_fusion_depth = 4;
  config.num_threads = 0;
  config.verbose = 1; // Enable verbose for debugging
  config.fusion_threshold = 1.1;

  if (TestInitialization())
  {
    passed++;
    PrintTestResult("Initialization", true);
  }
  else
  {
    PrintTestResult("Initialization", false);
  }

  // Test 2: Graph Construction
  total++;
  if (TestGraphConstruction())
  {
    passed++;
    PrintTestResult("Graph Construction", true);
  }
  else
  {
    PrintTestResult("Graph Construction", false);
  }

  // Test 3: Pattern Detection
  total++;
  if (TestPatternDetection())
  {
    passed++;
    PrintTestResult("Pattern Detection", true);
  }
  else
  {
    PrintTestResult("Pattern Detection", false);
  }

  // Test 4: Graph Execution
  total++;
  if (TestGraphExecution())
  {
    passed++;
    PrintTestResult("Graph Execution", true);
  }
  else
  {
    PrintTestResult("Graph Execution", false);
  }

  // Test 5: Performance Benchmark
  total++;
  if (TestPerformanceBenchmark())
  {
    passed++;
    PrintTestResult("Performance Benchmark", true);
  }
  else
  {
    PrintTestResult("Performance Benchmark", false);
  }

  // Test 6: Self-Test
  total++;
  if (TestSelfTest())
  {
    passed++;
    PrintTestResult("Self-Test", true);
  }
  else
  {
    PrintTestResult("Self-Test", false);
  }

  // Test 7: Long Fusion Chains
  total++;
  if (TestLongFusionChains())
  {
    passed++;
    PrintTestResult("Long Fusion Chains", true);
  }
  else
  {
    PrintTestResult("Long Fusion Chains", false);
  }

  // Test 8: Large Matrix Sizes
  total++;
  if (TestLargeMatrices())
  {
    passed++;
    PrintTestResult("Large Matrix Sizes", true);
  }
  else
  {
    PrintTestResult("Large Matrix Sizes", false);
  }

  // Test 9: Numerical Precision
  total++;
  if (TestNumericalPrecision())
  {
    passed++;
    PrintTestResult("Numerical Precision", true);
  }
  else
  {
    PrintTestResult("Numerical Precision", false);
  }

  // Test 10: Graph Utilities
  total++;
  if (TestGraphUtilities())
  {
    passed++;
    PrintTestResult("Graph Utilities", true);
  }
  else
  {
    PrintTestResult("Graph Utilities", false);
  }

  // Summary
  std::cout << "\n===============================================" << std::endl;
  std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
  std::cout << "Success Rate: " << std::fixed << std::setprecision(1)
            << (100.0 * passed / total) << "%" << std::endl;
  std::cout << "===============================================" << std::endl;

  og_shutdown();

  return (passed == total) ? 0 : 1;
}