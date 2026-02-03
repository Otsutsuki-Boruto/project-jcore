#include <chrono>
#include <cstdio>
#include "neural_primitives.h"

/**
 * The file effectively exists to test the project performance by executing multiple tests and measuring its performance.
 */


static void init_tensor_deterministic(npl_tensor_t &tensor) {
    float *data = static_cast<float*>(tensor.data);
    size_t n = tensor.size_bytes / sizeof(float);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i % 256) / 255.0f; // deterministic
    }
}

/* Calculate Statistics */
static void print_perf_stats(const char *name, const npl_perf_stats_t &stats, double gflops) {
    printf("%s:\n", name);
    printf("  GFLOPS: %.2f\n", gflops);
    printf("  Backend: %s\n", stats.backend_used);
    if (stats.was_fused) {
        printf("  FUSED: %zu ops\n", stats.operations_fused);
    }
    printf("\n");
}

/* main method */
int main() {

    /* Initialize Neural Layer With Default Config */
    printf("\n/*********************NPL Initialization*********************/\n");
    npl_config_t config;
    int ret = npl_get_default_config(&config);

    if (ret != NPL_OK) {
        printf("Failed to get default config\n");
        return 1;
    }

    int status = npl_init(&config);
    printf("NPL Initialization %s\n\n", status == NPL_OK ? " Successful" : "Failed");

    /* ============= Tests =================== */

    /* Matrix Multiplication Config */
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
