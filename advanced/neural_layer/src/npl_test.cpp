// advanced/neural_layer/src/NPL_test.cpp

#include "neural_primitives.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>

/* ========================================================================== */
/* Test Utilities                                                              */
/* ========================================================================== */

#define TEST_ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      fprintf(stderr, "[FAIL] %s: %s\n", __func__, msg); \
      return -1; \
    } \
  } while (0)

#define TEST_PASS(name) \
  do { \
    printf("[PASS] %s\n", name); \
  } while (0)

static int g_tests_passed = 0;
static int g_tests_failed = 0;

static void print_stats(const char *op, const npl_perf_stats_t *stats) {
  printf("  %s: %.3f ms, %.2f GFLOPS", op, stats->elapsed_ms, stats->gflops);
  if (stats->was_fused) {
    printf(" [FUSED: %zu ops]", stats->operations_fused);
  }
  printf("\n");
}

static bool tensors_equal(const float *a, const float *b, size_t n,
                         float tolerance = 1e-4f) {
  for (size_t i = 0; i < n; ++i) {
    float diff = std::abs(a[i] - b[i]);
    if (diff > tolerance) {
      fprintf(stderr, "Mismatch at index %zu: %.6f vs %.6f (diff: %.6f)\n",
              i, a[i], b[i], diff);
      return false;
    }
  }
  return true;
}

/* ========================================================================== */
/* Test: Initialization                                                        */
/* ========================================================================== */

static int test_initialization() {
  printf("\n=== Testing Initialization ===\n");

  // Test default config
  npl_config_t config;
  int ret = npl_get_default_config(&config);
  TEST_ASSERT(ret == NPL_OK, "Failed to get default config");

  // Initialize with default config
  ret = npl_init(&config);
  TEST_ASSERT(ret == NPL_OK, "Failed to initialize NPL");
  TEST_ASSERT(npl_is_initialized() == 1, "NPL not marked as initialized");

  // // Print system info
  // printf("\n%s\n", NPL_get_system_info());

  TEST_PASS("Initialization");
  return 0;
}

/* ========================================================================== */
/* Test: Tensor Operations                                                     */
/* ========================================================================== */

static int test_tensor_creation() {
  printf("\n=== Testing Tensor Creation ===\n");

  size_t shape[] = {2, 3, 4, 4};
  npl_tensor_t tensor;

  int ret = npl_create_tensor(nullptr, 4, shape, NPL_DTYPE_FP32,
                                NPL_LAYOUT_NCHW, &tensor);
  TEST_ASSERT(ret == NPL_OK, "Failed to create tensor");
  TEST_ASSERT(tensor.ndim == 4, "Incorrect ndim");
  TEST_ASSERT(tensor.shape[0] == 2, "Incorrect batch size");
  TEST_ASSERT(tensor.size_bytes == 2 * 3 * 4 * 4 * 4, "Incorrect size");

  ret = npl_allocate_tensor(&tensor);
  TEST_ASSERT(ret == NPL_OK, "Failed to allocate tensor");
  TEST_ASSERT(tensor.data != nullptr, "Tensor data is NULL");

  npl_free_tensor(&tensor);
  TEST_ASSERT(tensor.data == nullptr, "Tensor data not freed");

  TEST_PASS("Tensor Creation");
  return 0;
}

static int test_matmul() {
  printf("\n=== Testing Matrix Multiplication ===\n");

  const size_t M = 32, N = 32, K = 32;
  size_t shape_a[] = {M, K};
  size_t shape_b[] = {K, N};
  size_t shape_c[] = {M, N};

  npl_tensor_t A, B, C;
  npl_create_tensor(nullptr, 2, shape_a, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &A);
  npl_create_tensor(nullptr, 2, shape_b, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &B);
  npl_create_tensor(nullptr, 2, shape_c, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &C);

  npl_allocate_tensor(&A);
  npl_allocate_tensor(&B);
  npl_allocate_tensor(&C);

  // Initialize matrices
  float *a_data = static_cast<float*>(A.data);
  float *b_data = static_cast<float*>(B.data);
  for (size_t i = 0; i < M * K; ++i) a_data[i] = 1.0f;
  for (size_t i = 0; i < K * N; ++i) b_data[i] = 1.0f;

  // Perform MatMul
  npl_perf_stats_t stats = {};
  int ret = npl_matmul(&A, &B, &C, 1.0f, 0.0f, &stats);
  TEST_ASSERT(ret == NPL_OK, "MatMul failed");

  // Verify result (should be all K)
  float *c_data = static_cast<float*>(C.data);
  for (size_t i = 0; i < M * N; ++i) {
    TEST_ASSERT(std::abs(c_data[i] - K) < 0.001f, "MatMul result incorrect");
  }

  print_stats("MatMul", &stats);

  npl_free_tensor(&A);
  npl_free_tensor(&B);
  npl_free_tensor(&C);

  TEST_PASS("Matrix Multiplication");
  return 0;
}

static int test_elementwise_ops() {
  printf("\n=== Testing Element-wise Operations ===\n");

  const size_t N = 1024;
  size_t shape[] = {N};

  npl_tensor_t A, B, C;
  npl_create_tensor(nullptr, 1, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &A);
  npl_create_tensor(nullptr, 1, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &B);
  npl_create_tensor(nullptr, 1, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &C);

  npl_allocate_tensor(&A);
  npl_allocate_tensor(&B);
  npl_allocate_tensor(&C);

  float *a_data = static_cast<float*>(A.data);
  float *b_data = static_cast<float*>(B.data);
  float *c_data = static_cast<float*>(C.data);

  for (size_t i = 0; i < N; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = 2.0f;
  }

  // Test addition
  int ret = npl_add(&A, &B, &C, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Add failed");
  for (size_t i = 0; i < N; ++i) {
    TEST_ASSERT(std::abs(c_data[i] - (i + 2.0f)) < 0.001f,
                "Add result incorrect");
  }

  // Test multiplication
  ret = npl_mul(&A, &B, &C, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Mul failed");
  for (size_t i = 0; i < N; ++i) {
    TEST_ASSERT(std::abs(c_data[i] - (i * 2.0f)) < 0.001f,
                "Mul result incorrect");
  }

  npl_free_tensor(&A);
  npl_free_tensor(&B);
  npl_free_tensor(&C);

  TEST_PASS("Element-wise Operations");
  return 0;
}

/* ========================================================================== */
/* Test: Convolution                                                           */
/* ========================================================================== */

static int test_conv2d() {
  printf("\n=== Testing Conv2D ===\n");

  // Small test: 1 batch, 3 in channels, 8x8 image, 2 out channels, 3x3 kernel
  size_t in_shape[] = {1, 3, 8, 8};
  size_t w_shape[] = {2, 3, 3, 3};
  size_t bias_shape[] = {2};
  size_t out_shape[] = {1, 2, 6, 6}; // With stride=1, no padding

  npl_tensor_t input, weights, bias, output;
  npl_create_tensor(nullptr, 4, in_shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &input);
  npl_create_tensor(nullptr, 4, w_shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &weights);
  npl_create_tensor(nullptr, 1, bias_shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &bias);
  npl_create_tensor(nullptr, 4, out_shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &output);

  npl_allocate_tensor(&input);
  npl_allocate_tensor(&weights);
  npl_allocate_tensor(&bias);
  npl_allocate_tensor(&output);

  // Initialize with ones
  float *in_data = static_cast<float*>(input.data);
  float *w_data = static_cast<float*>(weights.data);
  float *b_data = static_cast<float*>(bias.data);

  for (size_t i = 0; i < 1 * 3 * 8 * 8; ++i) in_data[i] = 1.0f;
  for (size_t i = 0; i < 2 * 3 * 3 * 3; ++i) w_data[i] = 1.0f;
  for (size_t i = 0; i < 2; ++i) b_data[i] = 0.5f;

  // Convolution parameters
  npl_conv_params_t params = {};
  params.kernel_h = 3;
  params.kernel_w = 3;
  params.stride_h = 1;
  params.stride_w = 1;
  params.pad_h = 0;
  params.pad_w = 0;
  params.dilation_h = 1;
  params.dilation_w = 1;
  params.groups = 1;
  params.padding = NPL_PAD_VALID;
  params.activation = NPL_ACT_NONE;
  params.use_bias = 1;

  npl_perf_stats_t stats = {};
  int ret = npl_conv2d(&input, &weights, &bias, &params, &output, &stats);
  TEST_ASSERT(ret == NPL_OK, "Conv2D failed");

  // Verify result: each output should be 3*3*3 + 0.5 = 27.5
  float *out_data = static_cast<float*>(output.data);
  for (size_t i = 0; i < 1 * 2 * 6 * 6; ++i) {
    TEST_ASSERT(std::abs(out_data[i] - 27.5f) < 0.01f,
                "Conv2D result incorrect");
  }

  print_stats("Conv2D", &stats);

  npl_free_tensor(&input);
  npl_free_tensor(&weights);
  npl_free_tensor(&bias);
  npl_free_tensor(&output);

  TEST_PASS("Conv2D");
  return 0;
}

/* ========================================================================== */
/* Test: Activation Functions                                                  */
/* ========================================================================== */

static int test_activations() {
  printf("\n=== Testing Activation Functions ===\n");

  const size_t N = 1024;
  size_t shape[] = {N};

  npl_tensor_t input, output;
  npl_create_tensor(nullptr, 1, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &input);
  npl_create_tensor(nullptr, 1, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &output);

  npl_allocate_tensor(&input);
  npl_allocate_tensor(&output);

  float *in_data = static_cast<float*>(input.data);
  float *out_data = static_cast<float*>(output.data);

  // Initialize with range [-2, 2]
  for (size_t i = 0; i < N; ++i) {
    in_data[i] = -2.0f + 4.0f * i / N;
  }

  // Test ReLU
  int ret = npl_relu(&input, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "ReLU failed");
  for (size_t i = 0; i < N; ++i) {
    float expected = (in_data[i] > 0.0f) ? in_data[i] : 0.0f;
    TEST_ASSERT(std::abs(out_data[i] - expected) < 0.001f,
                "ReLU result incorrect");
  }

  // Test Sigmoid
  ret = npl_activation(&input, NPL_ACT_SIGMOID, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Sigmoid failed");
  for (size_t i = 0; i < N; ++i) {
    TEST_ASSERT(out_data[i] >= 0.0f && out_data[i] <= 1.0f,
                "Sigmoid out of range");
  }

  // Test Tanh
  ret = npl_activation(&input, NPL_ACT_TANH, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Tanh failed");
  for (size_t i = 0; i < N; ++i) {
    TEST_ASSERT(out_data[i] >= -1.0f && out_data[i] <= 1.0f,
                "Tanh out of range");
  }

  // Test GELU
  ret = npl_activation(&input, NPL_ACT_GELU, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "GELU failed");

  npl_free_tensor(&input);
  npl_free_tensor(&output);

  TEST_PASS("Activation Functions");
  return 0;
}

static int test_softmax() {
  printf("\n=== Testing Softmax ===\n");

  size_t shape[] = {2, 10}; // 2 batches, 10 classes
  npl_tensor_t input, output;
  npl_create_tensor(nullptr, 2, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &input);
  npl_create_tensor(nullptr, 2, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &output);

  npl_allocate_tensor(&input);
  npl_allocate_tensor(&output);

  float *in_data = static_cast<float*>(input.data);
  float *out_data = static_cast<float*>(output.data);

  // Initialize with random-like values
  for (size_t i = 0; i < 2 * 10; ++i) {
    in_data[i] = static_cast<float>(i % 10);
  }

  int ret = npl_softmax(&input, 1, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Softmax failed");

  // Verify sum to 1
  for (size_t batch = 0; batch < 2; ++batch) {
    float sum = 0.0f;
    for (size_t i = 0; i < 10; ++i) {
      sum += out_data[batch * 10 + i];
    }
    TEST_ASSERT(std::abs(sum - 1.0f) < 0.001f, "Softmax doesn't sum to 1");
  }

  npl_free_tensor(&input);
  npl_free_tensor(&output);

  TEST_PASS("Softmax");
  return 0;
}

/* ========================================================================== */
/* Test: Normalization                                                         */
/* ========================================================================== */

static int test_batch_norm() {
  printf("\n=== Testing Batch Normalization ===\n");

  size_t shape[] = {2, 3, 4, 4}; // 2 batches, 3 channels, 4x4 spatial
  npl_tensor_t input, output;
  npl_create_tensor(nullptr, 4, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &input);
  npl_create_tensor(nullptr, 4, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &output);

  npl_allocate_tensor(&input);
  npl_allocate_tensor(&output);

  float *in_data = static_cast<float*>(input.data);
  for (size_t i = 0; i < 2 * 3 * 4 * 4; ++i) {
    in_data[i] = static_cast<float>(i % 10);
  }

  // Batch norm parameters
  float mean[] = {5.0f, 5.0f, 5.0f};
  float variance[] = {4.0f, 4.0f, 4.0f};
  float gamma[] = {1.0f, 1.0f, 1.0f};
  float beta[] = {0.0f, 0.0f, 0.0f};

  npl_batchnorm_params_t params = {};
  params.mean = mean;
  params.variance = variance;
  params.gamma = gamma;
  params.beta = beta;
  params.epsilon = 1e-5f;
  params.training = 0;
  params.momentum = 0.1f;

  int ret = npl_batch_norm(&input, &params, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Batch norm failed");

  // Basic sanity check - output should be normalized
  float *out_data = static_cast<float*>(output.data);
  bool has_negative = false;
  bool has_positive = false;
  for (size_t i = 0; i < 2 * 3 * 4 * 4; ++i) {
    if (out_data[i] < 0) has_negative = true;
    if (out_data[i] > 0) has_positive = true;
  }
  TEST_ASSERT(has_negative && has_positive, "Batch norm output looks wrong");

  npl_free_tensor(&input);
  npl_free_tensor(&output);

  TEST_PASS("Batch Normalization");
  return 0;
}

static int test_layer_norm() {
  printf("\n=== Testing Layer Normalization ===\n");

  size_t shape[] = {4, 128}; // 4 sequences, 128 features
  npl_tensor_t input, output;
  npl_create_tensor(nullptr, 2, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &input);
  npl_create_tensor(nullptr, 2, shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &output);

  npl_allocate_tensor(&input);
  npl_allocate_tensor(&output);

  float *in_data = static_cast<float*>(input.data);
  for (size_t i = 0; i < 4 * 128; ++i) {
    in_data[i] = static_cast<float>(i % 20) - 10.0f;
  }

  npl_layernorm_params_t params = {};
  params.gamma = nullptr; // Will default to 1.0
  params.beta = nullptr;  // Will default to 0.0
  params.epsilon = 1e-5f;
  params.normalized_shape = 128;

  int ret = npl_layer_norm(&input, &params, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Layer norm failed");

  npl_free_tensor(&input);
  npl_free_tensor(&output);

  TEST_PASS("Layer Normalization");
  return 0;
}

/* ========================================================================== */
/* Test: Pooling                                                               */
/* ========================================================================== */

static int test_pooling() {
  printf("\n=== Testing Pooling Operations ===\n");

  size_t in_shape[] = {1, 2, 8, 8};
  size_t out_shape[] = {1, 2, 4, 4}; // 2x2 pooling with stride 2

  npl_tensor_t input, output;
  npl_create_tensor(nullptr, 4, in_shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &input);
  npl_create_tensor(nullptr, 4, out_shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &output);

  npl_allocate_tensor(&input);
  npl_allocate_tensor(&output);

  float *in_data = static_cast<float*>(input.data);
  for (size_t i = 0; i < 1 * 2 * 8 * 8; ++i) {
    in_data[i] = static_cast<float>(i % 16);
  }

  // Max pooling
  npl_pooling_params_t params = {};
  params.kernel_h = 2;
  params.kernel_w = 2;
  params.stride_h = 2;
  params.stride_w = 2;
  params.pad_h = 0;
  params.pad_w = 0;
  params.type = NPL_POOL_MAX;
  params.count_include_pad = 0;

  int ret = npl_pooling(&input, &params, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Max pooling failed");

  // Average pooling
  params.type = NPL_POOL_AVG;
  ret = npl_pooling(&input, &params, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Avg pooling failed");

  npl_free_tensor(&input);
  npl_free_tensor(&output);

  TEST_PASS("Pooling Operations");
  return 0;
}

static int test_global_avg_pool() {
  printf("\n=== Testing Global Average Pooling ===\n");

  size_t in_shape[] = {2, 64, 7, 7};
  size_t out_shape[] = {2, 64, 1, 1};

  npl_tensor_t input, output;
  npl_create_tensor(nullptr, 4, in_shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &input);
  npl_create_tensor(nullptr, 4, out_shape, NPL_DTYPE_FP32,
                     NPL_LAYOUT_NCHW, &output);

  npl_allocate_tensor(&input);
  npl_allocate_tensor(&output);

  float *in_data = static_cast<float*>(input.data);
  for (size_t i = 0; i < 2 * 64 * 7 * 7; ++i) {
    in_data[i] = 1.0f;
  }

  int ret = npl_global_avg_pool(&input, &output, nullptr);
  TEST_ASSERT(ret == NPL_OK, "Global avg pool failed");

  // Verify output is 1.0 (average of all 1.0s)
  float *out_data = static_cast<float*>(output.data);
  for (size_t i = 0; i < 2 * 64; ++i) {
    TEST_ASSERT(std::abs(out_data[i] - 1.0f) < 0.001f,
                "Global avg pool incorrect");
  }

  npl_free_tensor(&input);
  npl_free_tensor(&output);

  TEST_PASS("Global Average Pooling");
  return 0;
}

/* ========================================================================== */
/* Performance Benchmark                                                       */
/* ========================================================================== */

static void init_tensor_deterministic(npl_tensor_t &tensor) {
    float *data = static_cast<float*>(tensor.data);
    size_t n = tensor.size_bytes / sizeof(float);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i % 256) / 255.0f; // deterministic
    }
}

static void print_perf_stats(const char *name, const npl_perf_stats_t &stats, double gflops) {
    printf("%s:\n", name);
    printf("  GFLOPS: %.2f\n", gflops);
    printf("  Backend: %s\n", stats.backend_used);
    if (stats.was_fused) {
        printf("  FUSED: %zu ops\n", stats.operations_fused);
    }
    printf("\n");
}

static int benchmark_operations() {
    printf("\n=== Performance Benchmarks ===\n");

    /* ---------------- MatMul Benchmarks ---------------- */
    struct MatMulConfig {
        size_t M, N, K;
        int iterations;
    };

    MatMulConfig matmul_configs[] = {
      {64, 64, 64, 50},
      {256, 256, 256, 20},
        {512, 512, 512, 20},
        {1024, 1024, 1024, 20},
        {2048, 2048, 2048, 5},
        {4096, 4096, 4096, 2},
      {8192, 8192, 8192, 1}
    };

    for (const auto &cfg : matmul_configs) {
        size_t shape_a[] = {cfg.M, cfg.K};
        size_t shape_b[] = {cfg.K, cfg.N};
        size_t shape_c[] = {cfg.M, cfg.N};

        npl_tensor_t A, B, C;
        npl_create_tensor(nullptr, 2, shape_a, NPL_DTYPE_FP32, NPL_LAYOUT_NCHW, &A);
        npl_create_tensor(nullptr, 2, shape_b, NPL_DTYPE_FP32, NPL_LAYOUT_NCHW, &B);
        npl_create_tensor(nullptr, 2, shape_c, NPL_DTYPE_FP32, NPL_LAYOUT_NCHW, &C);

        npl_allocate_tensor(&A);
        npl_allocate_tensor(&B);
        npl_allocate_tensor(&C);

        init_tensor_deterministic(A);
        init_tensor_deterministic(B);

        // Warmup
        npl_perf_stats_t warmup_stats = {};
        npl_matmul(&A, &B, &C, 1.0f, 0.0f, &warmup_stats);

        // Timed iterations
        npl_perf_stats_t stats = {};
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cfg.iterations; ++i) {
            npl_matmul(&A, &B, &C, 1.0f, 0.0f, &stats);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double avg_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * cfg.iterations);
        double flops = 2.0 * cfg.M * cfg.N * cfg.K;
        double gflops = (flops / 1e9) / (avg_ms / 1000.0);

        char name_buf[128];
        snprintf(name_buf, sizeof(name_buf), "MatMul [%zu x %zu x %zu]", cfg.M, cfg.K, cfg.N);
        print_perf_stats(name_buf, stats, gflops);

        npl_free_tensor(&A);
        npl_free_tensor(&B);
        npl_free_tensor(&C);
    }

    /* ---------------- Conv2D Benchmarks ---------------- */
    struct ConvConfig {
        size_t batch, in_c, h, w, out_c, k;
        const char* name;
    };

    ConvConfig conv_configs[] = {
        {1, 64, 56, 56, 64, 3, "ResNet-50 Layer (small batch)"},
        {8, 64, 56, 56, 128, 3, "ResNet-50 Layer (batch=8)"},
        {1, 128, 28, 28, 256, 3, "ResNet-50 Mid Layer"},
        {4, 256, 14, 14, 512, 3, "ResNet-50 Deep Layer"}
    };

    for (const auto &cfg : conv_configs) {
        size_t out_h = cfg.h - cfg.k + 1;
        size_t out_w = cfg.w - cfg.k + 1;

        size_t in_shape[] = {cfg.batch, cfg.in_c, cfg.h, cfg.w};
        size_t w_shape[] = {cfg.out_c, cfg.in_c, cfg.k, cfg.k};
        size_t bias_shape[] = {cfg.out_c};
        size_t out_shape[] = {cfg.batch, cfg.out_c, out_h, out_w};

        npl_tensor_t input, weights, bias, output;
        npl_create_tensor(nullptr, 4, in_shape, NPL_DTYPE_FP32, NPL_LAYOUT_NCHW, &input);
        npl_create_tensor(nullptr, 4, w_shape, NPL_DTYPE_FP32, NPL_LAYOUT_NCHW, &weights);
        npl_create_tensor(nullptr, 1, bias_shape, NPL_DTYPE_FP32, NPL_LAYOUT_NCHW, &bias);
        npl_create_tensor(nullptr, 4, out_shape, NPL_DTYPE_FP32, NPL_LAYOUT_NCHW, &output);

        npl_allocate_tensor(&input);
        npl_allocate_tensor(&weights);
        npl_allocate_tensor(&bias);
        npl_allocate_tensor(&output);

        init_tensor_deterministic(input);
        init_tensor_deterministic(weights);
        init_tensor_deterministic(bias);

        npl_conv_params_t params = {};
        params.kernel_h = cfg.k;
        params.kernel_w = cfg.k;
        params.stride_h = 1;
        params.stride_w = 1;
        params.pad_h = 0;
        params.pad_w = 0;
        params.dilation_h = 1;
        params.dilation_w = 1;
        params.groups = 1;
        params.padding = NPL_PAD_VALID;
        params.activation = NPL_ACT_NONE;
        params.use_bias = 1;

        // Warmup
        npl_perf_stats_t warmup_stats = {};
        npl_conv2d(&input, &weights, &bias, &params, &output, &warmup_stats);

        // Timed run
        npl_perf_stats_t stats = {};
        auto start = std::chrono::high_resolution_clock::now();
        npl_conv2d(&input, &weights, &bias, &params, &output, &stats);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        double flops = 2.0 * cfg.batch * cfg.out_c * cfg.in_c * out_h * out_w * cfg.k * cfg.k;
        double gflops = (flops / 1e9) / (time_ms / 1000.0);

        print_perf_stats(cfg.name, stats, gflops);

        npl_free_tensor(&input);
        npl_free_tensor(&weights);
        npl_free_tensor(&bias);
        npl_free_tensor(&output);
    }

    return 0;
}


/* ========================================================================== */
/* Self Test (Comprehensive)                                                   */
/* ========================================================================== */

int npl_self_test(int verbose) {
  printf("\n");
  printf("╔════════════════════════════════════════════════════════════╗\n");
  printf("║   Neural Primitives Layer - Comprehensive Test       ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n");

  g_tests_passed = 0;
  g_tests_failed = 0;

  // Run all tests
  struct {
    const char *name;
    int (*test_fn)();
  } tests[] = {
      {"Initialization", test_initialization},
      {"Tensor Creation", test_tensor_creation},
      {"Matrix Multiplication", test_matmul},
      {"Element-wise Operations", test_elementwise_ops},
      {"Conv2D", test_conv2d},
      {"Activation Functions", test_activations},
      {"Softmax", test_softmax},
      {"Batch Normalization", test_batch_norm},
      {"Layer Normalization", test_layer_norm},
      {"Pooling Operations", test_pooling},
      {"Global Avg Pooling", test_global_avg_pool},
  };

  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    if (verbose) {
      printf("\n=== Running: %s ===\n", tests[i].name);
    }

    int result = tests[i].test_fn();
    if (result == 0) {
      g_tests_passed++;
    } else {
      g_tests_failed++;
      if (!verbose) {
        printf("[FAIL] %s\n", tests[i].name);
      }
    }
  }

  // Run benchmarks if verbose
  if (verbose) {
    benchmark_operations();
  }

  // Summary
  printf("\n");
  printf("╔════════════════════════════════════════════════════════════╗\n");
  printf("║                      Test Summary                          ║\n");
  printf("╠════════════════════════════════════════════════════════════╣\n");
  printf("║  Total Tests:  %3d                                         ║\n", g_tests_passed + g_tests_failed);
  printf("║  Passed:       %3d                                         ║\n", g_tests_passed);
  printf("║  Failed:       %3d                                         ║\n", g_tests_failed);
  printf("╚════════════════════════════════════════════════════════════╝\n");

  return (g_tests_failed == 0) ? NPL_OK : -1;
}

/* ========================================================================== */
/* Main Test Entry Point                                                       */
/* ========================================================================== */

int main(int argc, char **argv) {
  int verbose = 1; // Default verbose

  if (argc > 1) {
    if (strcmp(argv[1], "-q") == 0 || strcmp(argv[1], "--quiet") == 0) {
      verbose = 0;
    }
  }

  int result = npl_self_test(verbose);

  // Cleanup
  if (npl_is_initialized()) {
    npl_shutdown();
  }

  return (result == NPL_OK) ? 0 : 1;
}