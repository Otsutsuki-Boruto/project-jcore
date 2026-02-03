// kernels_openblas.cpp
// OpenBLAS SGEMM wrapper for Adaptive Kernel Auto-Tuner

#include "kernels_openblas.h"
#include <openblas/cblas.h>
#include <cstddef>

// OpenBLAS SGEMM wrapper
// Computes C = A * B where all matrices are row-major
// A is M x K, B is K x N, C is M x N
extern "C" void openblas_sgemm(
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t K)
{
  // OpenBLAS cblas_sgemm parameters:
  // Order: CblasRowMajor for row-major matrices
  // TransA, TransB: CblasNoTrans (no transpose)
  // M, N, K: matrix dimensions
  // alpha = 1.0, beta = 0.0 for C = A * B
  // lda = K (leading dimension of A in row-major)
  // ldb = N (leading dimension of B in row-major)
  // ldc = N (leading dimension of C in row-major)

  cblas_sgemm(
      CblasRowMajor, // Matrix order
      CblasNoTrans,  // Don't transpose A
      CblasNoTrans,  // Don't transpose B
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(K),
      1.0f, // alpha
      A,
      static_cast<int>(K), // lda
      B,
      static_cast<int>(N), // ldb
      0.0f,                // beta (C is overwritten)
      C,
      static_cast<int>(N) // ldc
  );
}