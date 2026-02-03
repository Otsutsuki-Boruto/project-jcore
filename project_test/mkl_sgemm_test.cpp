#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mkl.h>

int main() {
    const int N = 8192;              // Matrix size
    const int ITERATIONS = 1;      // Set 2 for large matrices, higher for small matrices
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    float *A = (float*) mkl_malloc(N * N * sizeof(float), 64);
    float *B = (float*) mkl_malloc(N * N * sizeof(float), 64);
    float *C = (float*) mkl_malloc(N * N * sizeof(float), 64);

    if (!A || !B || !C) {
        printf("Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    // Warm-up
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, alpha, A, N, B, N, beta, C, N);

    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, alpha, A, N, B, N, beta, C, N);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed =
        (t1.tv_sec - t0.tv_sec) +
        (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    double flops_total = 2.0 * N * N * N * ITERATIONS;
    double gflops = flops_total / elapsed / 1e9;

    printf("MKL SGEMM %dx%d (%d iters): %.2f GFLOPS (%.3f s)\n",
           N, N, ITERATIONS, gflops, elapsed);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
