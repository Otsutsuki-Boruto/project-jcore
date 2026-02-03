// test_openblas_matmul.cpp
// Compile:

/*
g++ -O3 -march=native project_test/test_openblas_matmul.cpp \
-Iadvanced/include/ -Ladvanced/lib/static -lopenblas -o project_test/test_openblas_matmul
*/

//
// Notes:
// - Performs one DGEMM: C = A * B
// - Matrix size: 8192 x 8192
// - Measures wall-clock time and reports GFLOPS

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "openblas/cblas.h"

int main() {
    constexpr size_t N = 8192;
    constexpr double alpha = 1.0;
    constexpr double beta  = 0.0;

    // Allocate aligned memory for better performance
    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;

    if (posix_memalign(reinterpret_cast<void **>(&A), 64, sizeof(double) * N * N) ||
        posix_memalign(reinterpret_cast<void **>(&B), 64, sizeof(double) * N * N) ||
        posix_memalign(reinterpret_cast<void **>(&C), 64, sizeof(double) * N * N)) {
        std::fprintf(stderr, "Memory allocation failed\n");
        return 1;
        }

    // Initialize matrices
    for (size_t i = 0; i < N * N; ++i) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    // Warm-up run (important for fair timing)
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N, N, N,
        alpha,
        A, N,
        B, N,
        beta,
        C, N
    );

    std::memset(C, 0, sizeof(double) * N * N);

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N, N, N,
        alpha,
        A, N,
        B, N,
        beta,
        C, N
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Compute FLOPs: 2 * N^3
    double flops = 2.0 * static_cast<double>(N) * N * N;
    double gflops = (flops / elapsed.count()) / 1e9;

    std::printf("Matrix size        : %zu x %zu\n", N, N);
    std::printf("Elapsed time (s)   : %.6f\n", elapsed.count());
    std::printf("Performance (GFLOP/s): %.2f\n", gflops);

    std::printf("C[0,0] = %.2f, C[N-1,N-1] = %.2f\n", C[0], C[N*N-1]);

    free(A);
    free(B);
    free(C);

    return 0;
}

