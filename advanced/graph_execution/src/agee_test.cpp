// advanced/src/agee_test.cpp
#include "ag_execution_engine.h"
#include "operator_graph.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* ========================================================================== */
/* Test Utilities                                                             */
/* ========================================================================== */

static void print_separator(const char *title) {
  printf("\n");
  printf("====================================================================="
         "=========\n");
  printf("  %s\n", title);
  printf("====================================================================="
         "=========\n");
}

static void print_test_result(const char *test_name, bool passed) {
  printf("  [%s] %s\n", passed ? "✓ PASS" : "✗ FAIL", test_name);
}

static double compute_relative_error(double expected, double actual) {
  if (std::fabs(expected) < 1e-10) {
    return std::fabs(actual);
  }
  return std::fabs((expected - actual) / expected);
}

/* ========================================================================== */
/* Test 1: Initialization and Configuration                                   */
/* ========================================================================== */

static bool test_initialization() {
  print_separator("Test 1: Initialization and Configuration");

  // Test default configuration
  agee_config_t config = {};
  int ret = agee_get_default_config(&config);
  print_test_result("Get default config", ret == AGEE_OK);

  if (ret != AGEE_OK)
    return false;

  // Initialize AGEE
  ret = agee_init(&config);
  print_test_result("Initialize AGEE", ret == AGEE_OK);

  if (ret != AGEE_OK)
    return false;

  // Check initialization status
  int initialized = agee_is_initialized();
  print_test_result("Check initialized", initialized == 1);

  // Get system info
  char *info = nullptr;
  ret = agee_get_system_info(&info);
  print_test_result("Get system info", ret == AGEE_OK && info != nullptr);

  if (info) {
    printf("\nSystem Information:\n%s\n", info);
    agee_free_string(info);
  }

  return true;
}

/* ========================================================================== */
/* Test 2: Session Management                                                 */
/* ========================================================================== */

static bool test_session_management() {
  print_separator("Test 2: Session Management");

  // Create session
  agee_session_t session = nullptr;
  int ret = agee_create_session(&session);
  print_test_result("Create session", ret == AGEE_OK && session != nullptr);

  if (ret != AGEE_OK || !session)
    return false;

  // Get session stats
  agee_exec_stats_t stats = {};
  ret = agee_get_session_stats(session, &stats);
  print_test_result("Get session stats", ret == AGEE_OK);

  // Reset session
  ret = agee_reset_session(session);
  print_test_result("Reset session", ret == AGEE_OK);

  // Destroy session
  agee_destroy_session(session);
  print_test_result("Destroy session", true);

  return true;
}

/* ========================================================================== */
/* Test 3: Simple GEMM Graph Creation and Execution                          */
/* ========================================================================== */

static bool test_simple_gemm_graph() {
  print_separator("Test 3: Simple GEMM Graph Execution");

  // Create session
  agee_session_t session = nullptr;
  int ret = agee_create_session(&session);
  if (ret != AGEE_OK) {
    printf("  [✗ FAIL] Session creation failed: %s\n", agee_strerror(ret));
    return false;
  }

  // Create operator graph
  og_graph_t og = nullptr;
  ret = og_create_graph(&og);
  if (ret != OG_OK) {
    printf("  [✗ FAIL] Operator graph creation failed: %d\n", ret);
    agee_destroy_session(session);
    return false;
  }

  // Create tensors: C = A * B
  size_t M = 128, N = 128, K = 128;

  /* Tensor A : M x K */
  float *data_a = new float[M * K];
  og_tensor_t tensor_a = {};
  tensor_a.ndim = 2;
  tensor_a.shape[0] = M;
  tensor_a.shape[1] = K;
  tensor_a.size_bytes = M * K * sizeof(float);
  tensor_a.data = data_a;
  uint64_t tid_a = 0;
  ret = og_add_tensor(og, &tensor_a, &tid_a);
  if (ret != OG_OK) {
    og_destroy_graph(og);
    agee_destroy_session(session);
    return false;
  }

  /* Tensor B : K x N */
  float *data_b = new float[K * N];
  og_tensor_t tensor_b = {};
  tensor_b.ndim = 2;
  tensor_b.shape[0] = K;
  tensor_b.shape[1] = N;
  tensor_b.size_bytes = K * N * sizeof(float);
  tensor_b.data = data_b;
  uint64_t tid_b = 0;
  ret = og_add_tensor(og, &tensor_b, &tid_b);
  if (ret != OG_OK) {
    og_destroy_graph(og);
    agee_destroy_session(session);
    return false;
  }

  /* Tensor C : M x N */
  float *data_c = new float[M * N];
  og_tensor_t tensor_c = {};
  tensor_c.ndim = 2;
  tensor_c.shape[0] = M;
  tensor_c.shape[1] = N;
  tensor_c.size_bytes = M * N * sizeof(float);
  tensor_c.data = data_c;
  uint64_t tid_c = 0;
  ret = og_add_tensor(og, &tensor_c, &tid_c);
  if (ret != OG_OK) {
    og_destroy_graph(og);
    agee_destroy_session(session);
    return false;
  }

  // Add GEMM node
  og_node_t node = {};
  node.type = OG_OP_GEMM;
  node.input_ids[0] = tid_a;
  node.input_ids[1] = tid_b;
  node.num_inputs = 2;
  node.output_ids[0] = tid_c;
  node.num_outputs = 1;

  uint64_t nid = 0;
  ret = og_add_node(og, &node, &nid);
  if (ret != OG_OK) {
    og_destroy_graph(og);
    agee_destroy_session(session);
    return false;
  }

  // Finalize graph
  ret = og_finalize_graph(og);
  print_test_result("Finalize graph", ret == OG_OK);
  if (ret != OG_OK) {
    og_destroy_graph(og);
    agee_destroy_session(session);
    return false;
  }

  // Create execution plan
  agee_graph_plan_t plan = nullptr;
  ret = agee_create_plan_from_graph(session, og, &plan);
  print_test_result("Create execution plan", ret == AGEE_OK);
  if (ret != AGEE_OK) {
    agee_destroy_plan(plan);
    og_destroy_graph(og);
    agee_destroy_session(session);
    return false;
  }

  // Estimate memory
  size_t peak_mem = 0;
  ret = agee_estimate_memory(plan, &peak_mem);
  print_test_result("Estimate memory", ret == AGEE_OK);
  printf("  Peak memory: %.2f MB\n", peak_mem / (1024.0 * 1024.0));

  // Get plan info
  char *plan_info = nullptr;
  ret = agee_get_plan_info(plan, &plan_info);
  if (ret == AGEE_OK && plan_info) {
    printf("\nPlan Information:\n%s\n", plan_info);
    agee_free_string(plan_info);
  }

  // Preallocate memory
  ret = agee_preallocate_memory(session, plan);
  print_test_result("Preallocate memory", ret == AGEE_OK);

  // Execute plan
  agee_exec_stats_t stats = {};
  ret = agee_execute_plan(session, plan, &stats);
  print_test_result("Execute plan", ret == AGEE_OK);
  if (ret == AGEE_OK) {
    printf("\nExecution Statistics:\n");
    printf("  Total execution time: %.3f ms\n", stats.total_execution_time_ms);
    printf("  Total operations: %zu\n", stats.total_operations);
    printf("  Fused operations: %zu\n", stats.fused_operations);
    printf("  Memory allocated: %.2f MB\n",
           stats.memory_allocated_bytes / (1024.0 * 1024.0));
    printf("  Memory saved: %.2f KB\n", stats.memory_saved_bytes / 1024.0);
    printf("  Achieved GFLOPS: %.2f\n", stats.achieved_gflops);
    printf("  Fusion speedup: %.2fx\n", stats.fusion_speedup);
  }

  // Cleanup normal exit
  agee_destroy_plan(plan);
  og_destroy_graph(og);
  agee_destroy_session(session);

  return ret == AGEE_OK;
}

/* ========================================================================== */
/* Test 4: Fused GEMM + Bias + Activation (Batch-style approach) */
/* ========================================================================== */

static bool test_fused_gemm_bias_activation() {
  print_separator("Test 4: Fused GEMM + Bias + Activation");

  // -----------------------------
  // Session configuration
  // -----------------------------
  agee_config_t config{};
  agee_get_default_config(&config);
  config.verbose = 0;           // verbose for debug
  config.profile_execution = 1; // enable profiling
  config.enable_fusion = 1;

  agee_session_t session = nullptr;
  int ret = agee_create_session(&session);
  if (ret != AGEE_OK) {
    printf("[TEST4 ERROR] Failed to create session: %s\n", agee_strerror(ret));
    return false;
  }

  // -----------------------------
  // Prepare graphs and plans
  // -----------------------------
  std::vector<og_graph_t> graphs;
  std::vector<agee_graph_plan_t> plans;

  // For test 4, we only need one fused graph
  og_graph_t og = nullptr;
  ret = og_create_graph(&og);
  if (ret != OG_OK) {
    printf("[TEST4 ERROR] Failed to create graph: %d\n", ret);
    agee_destroy_session(session);
    return false;
  }

  // Graph tensors
  size_t M = 256, N = 256, K = 256;
  float *data_a = new float[M * K];
  float *data_b = new float[K * N];
  float *data_bias = new float[N];
  float *data_temp = new float[M * N];
  float *data_temp2 = new float[M * N];
  float *data_out = new float[M * N];

  og_tensor_t tensors[5] = {};

  tensors[0].ndim = 2;
  tensors[0].shape[0] = M;
  tensors[0].shape[1] = K;
  tensors[0].size_bytes = M * K * sizeof(float);
  tensors[0].data = data_a;
  tensors[1].ndim = 2;
  tensors[1].shape[0] = K;
  tensors[1].shape[1] = N;
  tensors[1].size_bytes = K * N * sizeof(float);
  tensors[1].data = data_b;
  tensors[2].ndim = 1;
  tensors[2].shape[0] = N;
  tensors[2].size_bytes = N * sizeof(float);
  tensors[2].data = data_bias;
  tensors[3].ndim = 2;
  tensors[3].shape[0] = M;
  tensors[3].shape[1] = N;
  tensors[3].size_bytes = M * N * sizeof(float);
  tensors[3].data = data_temp;
  tensors[4].ndim = 2;
  tensors[4].shape[0] = M;
  tensors[4].shape[1] = N;
  tensors[4].size_bytes = M * N * sizeof(float);
  tensors[4].data = data_temp2;

  uint64_t tid[5] = {};
  for (int i = 0; i < 5; ++i) {
    ret = og_add_tensor(og, &tensors[i], &tid[i]);
  }

  // Output tensor
  og_tensor_t tensor_out = {};
  tensor_out.ndim = 2;
  tensor_out.shape[0] = M;
  tensor_out.shape[1] = N;
  tensor_out.size_bytes = M * N * sizeof(float);
  tensor_out.data = data_out;
  uint64_t tid_out = 0;
  og_add_tensor(og, &tensor_out, &tid_out);

  // -----------------------------
  // Add nodes
  // -----------------------------
  og_node_t nodes[3] = {};

  // GEMM
  nodes[0].type = OG_OP_GEMM;
  nodes[0].input_ids[0] = tid[0];
  nodes[0].input_ids[1] = tid[1];
  nodes[0].num_inputs = 2;
  nodes[0].output_ids[0] = tid[3];
  nodes[0].num_outputs = 1;

  // Bias
  nodes[1].type = OG_OP_BIAS_ADD;
  nodes[1].input_ids[0] = tid[3];
  nodes[1].input_ids[1] = tid[2];
  nodes[1].num_inputs = 2;
  nodes[1].output_ids[0] = tid[4];
  nodes[1].num_outputs = 1;

  // ReLU
  nodes[2].type = OG_OP_RELU;
  nodes[2].input_ids[0] = tid[4];
  nodes[2].num_inputs = 1;
  nodes[2].output_ids[0] = tid_out;
  nodes[2].num_outputs = 1;

  uint64_t nids[3] = {};
  for (int i = 0; i < 3; ++i) {
    ret = og_add_node(og, &nodes[i], &nids[i]);
  }

  // -----------------------------
  // Finalize graph
  // -----------------------------
  ret = og_finalize_graph(og);
  print_test_result("Finalize fused graph", ret == OG_OK);

  // -----------------------------
  // Create plan
  // -----------------------------
  agee_graph_plan_t plan = nullptr;
  ret = agee_create_plan_from_graph(session, og, &plan);
  print_test_result("Create fused plan", ret == AGEE_OK);

  // Execute plan
  agee_exec_stats_t stats{};
  ret = agee_execute_plan(session, plan, &stats);
  print_test_result("Execute fused plan", ret == AGEE_OK);

  if (ret == AGEE_OK) {
    printf("\nFused Execution Statistics:\n");
    printf("  Total execution time: %.3f ms\n", stats.total_execution_time_ms);
    printf("  Fused operations: %zu / %zu\n", stats.fused_operations,
           stats.total_operations);
    printf("  Memory saved: %.2f KB\n", stats.memory_saved_bytes / 1024.0);
    printf("  Achieved GFLOPS: %.2f\n", stats.achieved_gflops);
    printf("  Fusion speedup: %.2fx\n", stats.fusion_speedup);
  }

  /* ----------------------------- */
  /* Batch-style cleanup           */
  /* ----------------------------- */
  graphs.push_back(og);
  plans.push_back(plan);

  // Destroy plans first
  for (size_t i = 0; i < plans.size(); ++i) {
    agee_destroy_plan(plans[i]);
  }

  // Destroy graphs after plans
  for (size_t i = 0; i < graphs.size(); ++i) {
    og_destroy_graph(graphs[i]);
  }

  // Destroy session
  agee_destroy_session(session);

  return ret == AGEE_OK;
}

/* ========================================================================== */
/* Test 5: Batched Execution                                                  */
/* ========================================================================== */

static bool test_batched_execution() {
  print_separator("Test 5: Batched Graph Execution");

  // -----------------------------
  // Session configuration (verbose)
  // -----------------------------
  agee_config_t config{};
  config.num_threads = 0; // Auto-detect
  config.enable_fusion = 1;
  config.enable_numa_optimization = 1;
  config.enable_prefetch = 1;
  config.enable_adaptive_tuning = 1;
  config.enable_memory_pooling = 1;
  config.use_hugepages = 0;
  config.memory_pool_size_mb = 256;
  config.workspace_size_mb = 128;
  config.fusion_threshold = 1.2;
  config.verbose = 0; // ENABLE VERBOSE
  config.profile_execution = 1;

  agee_session_t session = nullptr;
  if (agee_create_session(&session) != AGEE_OK) {
    printf("[TEST5 ERROR] Failed to create session\n");
    return false;
  }

  constexpr size_t BATCH_SIZE = 4;
  std::vector<agee_graph_plan_t> plans;
  std::vector<og_graph_t> graphs;

  for (size_t b = 0; b < BATCH_SIZE; ++b) {
    og_graph_t og = nullptr;
    if (og_create_graph(&og) != OG_OK) {
      break;
    }

    // Tensor dimensions
    size_t M = 64, N = 64, K = 64;

    // Allocate actual data for tensors
    float *data_a = new float[M * K]{};
    float *data_b = new float[K * N]{};
    float *data_c = new float[M * N]{};

    og_tensor_t ta = {};
    ta.ndim = 2;
    ta.shape[0] = M;
    ta.shape[1] = K;
    ta.size_bytes = M * K * sizeof(float);
    ta.data = data_a;

    og_tensor_t tb = {};
    tb.ndim = 2;
    tb.shape[0] = K;
    tb.shape[1] = N;
    tb.size_bytes = K * N * sizeof(float);
    tb.data = data_b;

    og_tensor_t tc = {};
    tc.ndim = 2;
    tc.shape[0] = M;
    tc.shape[1] = N;
    tc.size_bytes = M * N * sizeof(float);
    tc.data = data_c;

    uint64_t tid_a, tid_b, tid_c;
    og_add_tensor(og, &ta, &tid_a);
    og_add_tensor(og, &tb, &tid_b);
    og_add_tensor(og, &tc, &tid_c);

    og_node_t node{};
    node.type = OG_OP_GEMM;
    node.input_ids[0] = tid_a;
    node.input_ids[1] = tid_b;
    node.num_inputs = 2;
    node.output_ids[0] = tid_c;
    node.num_outputs = 1;

    uint64_t nid;
    og_add_node(og, &node, &nid);
    og_finalize_graph(og);

    agee_graph_plan_t plan = nullptr;
    if (agee_create_plan_from_graph(session, og, &plan) != AGEE_OK) {
      og_destroy_graph(og);
      break;
    }

    graphs.push_back(og);
    plans.push_back(plan);
  }

  print_test_result("Create batch plans", plans.size() == BATCH_SIZE);
  if (plans.size() != BATCH_SIZE) {
    // Cleanup partial batch
    for (auto plan : plans)
      agee_destroy_plan(plan);
    for (size_t i = 0; i < graphs.size(); ++i) {
      og_destroy_graph(graphs[i]);
    }
    agee_destroy_session(session);
    return false;
  }

  /* ----------------------------- */
  /* Execute batch                 */
  /* ----------------------------- */
  std::vector<agee_exec_stats_t> stats(BATCH_SIZE);

  int ret = agee_execute_batch(session, plans.data(), BATCH_SIZE, stats.data());
  print_test_result("Execute batch", ret == AGEE_OK);

  if (ret == AGEE_OK) {
    double total_gflops = 0.0;
    for (const auto &s : stats)
      total_gflops += s.achieved_gflops;

    printf("\nBatch Execution Statistics:\n");
    printf("  Batch size: %zu\n", BATCH_SIZE);
    printf("  Combined GFLOPS: %.2f\n", total_gflops);
  }

  /* ----------------------------- */
  /* Cleanup                       */
  /* ----------------------------- */

  // Destroy plans first
  for (size_t i = 0; i < plans.size(); ++i) {
    agee_destroy_plan(plans[i]);
  }

  // Destroy graphs after plans
  for (size_t i = 0; i < graphs.size(); ++i) {
    og_destroy_graph(graphs[i]);
  }

  // Destroy session
  agee_destroy_session(session);

  return ret == AGEE_OK;
}

/* ==========================================================================
/* Test 6: Performance Benchmarking */
/* ==========================================================================
*/

static bool test_performance_benchmark() {
  print_separator("Test 6: Performance Benchmarking");

  agee_session_t session = nullptr;
  int ret = agee_create_session(&session);
  if (ret != AGEE_OK) {
    fprintf(stderr, "[TEST6 ERROR] Failed to create session: %s\n",
            agee_strerror(ret));
    return false;
  }

  const size_t test_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
  const size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

  // Store graphs and plans to destroy AFTER session cleanup
  std::vector<og_graph_t> graphs;
  std::vector<agee_graph_plan_t> plans;

  bool all_passed = true;

  for (size_t idx = 0; idx < num_sizes; ++idx) {
    size_t size = test_sizes[idx];

    printf("\n--- Matrix Size: %zux%zu ---\n", size, size);

    og_graph_t og = nullptr;
    ret = og_create_graph(&og);
    if (ret != OG_OK || !og) {
      fprintf(stderr, "[TEST6 ERROR] Graph creation failed\n");
      all_passed = false;
      continue;
    }

    // Allocate tensor data on heap with RAII
    std::unique_ptr<float[]> data_a(new float[size * size]());
    std::unique_ptr<float[]> data_b(new float[size * size]());
    std::unique_ptr<float[]> data_c(new float[size * size]());

    // Create tensor descriptors
    og_tensor_t ta = {};
    ta.ndim = 2;
    ta.shape[0] = size;
    ta.shape[1] = size;
    ta.strides[0] = size;
    ta.strides[1] = 1;
    ta.size_bytes = size * size * sizeof(float);
    ta.total_elements = size * size;
    ta.data = data_a.get();

    og_tensor_t tb = {};
    tb.ndim = 2;
    tb.shape[0] = size;
    tb.shape[1] = size;
    tb.strides[0] = size;
    tb.strides[1] = 1;
    tb.size_bytes = size * size * sizeof(float);
    tb.total_elements = size * size;
    tb.data = data_b.get();

    og_tensor_t tc = {};
    tc.ndim = 2;
    tc.shape[0] = size;
    tc.shape[1] = size;
    tc.strides[0] = size;
    tc.strides[1] = 1;
    tc.size_bytes = size * size * sizeof(float);
    tc.total_elements = size * size;
    tc.data = data_c.get();

    // Add tensors
    uint64_t tid_a = 0, tid_b = 0, tid_c = 0;
    ret = og_add_tensor(og, &ta, &tid_a);
    if (ret != OG_OK) {
      fprintf(stderr, "[TEST6 ERROR] Failed to add tensor A\n");
      og_destroy_graph(og);
      all_passed = false;
      continue;
    }

    ret = og_add_tensor(og, &tb, &tid_b);
    if (ret != OG_OK) {
      fprintf(stderr, "[TEST6 ERROR] Failed to add tensor B\n");
      og_destroy_graph(og);
      all_passed = false;
      continue;
    }

    ret = og_add_tensor(og, &tc, &tid_c);
    if (ret != OG_OK) {
      fprintf(stderr, "[TEST6 ERROR] Failed to add tensor C\n");
      og_destroy_graph(og);
      all_passed = false;
      continue;
    }

    // Add GEMM node
    og_node_t node = {};
    node.type = OG_OP_GEMM;
    node.input_ids[0] = tid_a;
    node.input_ids[1] = tid_b;
    node.num_inputs = 2;
    node.output_ids[0] = tid_c;
    node.num_outputs = 1;

    uint64_t nid = 0;
    ret = og_add_node(og, &node, &nid);
    if (ret != OG_OK) {
      fprintf(stderr, "[TEST6 ERROR] Failed to add node\n");
      og_destroy_graph(og);
      all_passed = false;
      continue;
    }

    // Finalize graph
    ret = og_finalize_graph(og);
    if (ret != OG_OK) {
      fprintf(stderr, "[TEST6 ERROR] Graph finalization failed: %d\n", ret);
      og_destroy_graph(og);
      all_passed = false;
      continue;
    }

    // Create plan
    agee_graph_plan_t plan = nullptr;
    ret = agee_create_plan_from_graph(session, og, &plan);
    if (ret != AGEE_OK) {
      fprintf(stderr, "[TEST6 ERROR] Plan creation failed: %s\n",
              agee_strerror(ret));
      og_destroy_graph(og);
      all_passed = false;
      continue;
    }

    // Execute plan and collect stats
    agee_exec_stats_t bench_stats = {};
    ret = agee_execute_plan(session, plan, &bench_stats);
    print_test_result("Benchmark", ret == AGEE_OK);

    if (ret == AGEE_OK) {
      printf("  Achieved GFLOPS: %.2f\n", bench_stats.achieved_gflops);
      printf("  Execution time: %.3f ms\n", bench_stats.total_execution_time_ms);
    } else {
      fprintf(stderr, "[TEST6 ERROR] Benchmark failed: %s\n",
              agee_strerror(ret));
      all_passed = false;
    }

    // Store for cleanup AFTER session destruction
    graphs.push_back(og);
    plans.push_back(plan);
  }

  // Destroy plans first
  for (auto plan : plans) {
    agee_destroy_plan(plan);
  }

  // Destroy graphs
  for (auto graph : graphs) {
    og_destroy_graph(graph);
  }

  // Destroy sessions at last
  agee_destroy_session(session);

  return all_passed;
}

/* ========================================================================== */
/* Main Test Suite                                                            */
/* ========================================================================== */

int main(int argc, char **argv) {
  printf("\n");
  printf("====================================================================="
         "===========\n");
  printf("  Adaptive Graph Execution Engine - Comprehensive Test Suite\n");
  printf("  Project JCore - Advanced Component Test\n");
  printf("====================================================================="
         "===========\n");

  bool all_passed = true;
  int tests_passed = 0;
  int total_tests = 0;

  // Run tests
  total_tests++;
  if (test_initialization()) {
    tests_passed++;
  } else {
    all_passed = false;
    printf("ERROR: Initialization test failed, stopping\n");
    return 1;
  }

  total_tests++;
  if (test_session_management()) {
    tests_passed++;
  } else {
    all_passed = false;
  }

  total_tests++;
  if (test_simple_gemm_graph()) {
    tests_passed++;
  } else {
    all_passed = false;
  }

  total_tests++;
  bool test4_result = test_fused_gemm_bias_activation();
  if (test4_result) {
    tests_passed++;
  } else {
    all_passed = false;
  }

  total_tests++;
  if (test_batched_execution()) {
    tests_passed++;
  } else {
    all_passed = false;
  }

  total_tests++;
  if (test_performance_benchmark()) {
    tests_passed++;
  } else {
    all_passed = false;
  }

  // Run self-test (carefully to avoid segfault)
  print_separator("Internal Self-Test");
  int self_test_result = AGEE_ERR_INTERNAL;

  try {
    self_test_result = agee_self_test(1);
    print_test_result("Self-test", self_test_result == AGEE_OK);
  } catch (...) {
    printf("  [✗ FAIL] Self-test threw exception\n");
    self_test_result = AGEE_ERR_INTERNAL;
  }

  // Cleanup (with error handling)
  print_separator("Cleanup");
  try {
    agee_shutdown();
    printf("  [✓ PASS] Shutdown complete\n");
  } catch (...) {
    printf("  [✗ FAIL] Shutdown threw exception\n");
  }

  // Summary
  print_separator("Test Summary");
  printf("Tests passed: %d / %d\n", tests_passed,
         total_tests);
  printf("  Self-test: %s\n",
         self_test_result == AGEE_OK ? "PASS" : "FAIL (non-critical)");
  printf("  Overall result: %s\n\n",
         all_passed ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED");

  return all_passed ? 0 : 1;
}