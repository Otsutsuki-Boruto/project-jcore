// mock_kernels.cpp
// Three lightweight, well-documented naive matmul implementations that act as
// stand-ins for OpenBLAS/BLIS/LIBXSMM kernels for testing and registration.
// They match the signature `void kernel(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)`.
//
// Each is declared "extern" style to be registered via jcore_register_impl.

#include <stddef.h>
#include <string.h>
#include <stdio.h>

#include "adaptive_tuner.h" /* for the signature typedef used in headers */

/* naive single-threaded i-j-k order simple matmul */
static void naive_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
  if (!A || !B || !C)
    return;
  /* zero C */
  for (size_t i = 0; i < M * N; ++i)
    C[i] = 0.0f;

  for (size_t i = 0; i < M; ++i)
  {
    for (size_t k = 0; k < K; ++k)
    {
      float a = A[i * K + k];
      for (size_t j = 0; j < N; ++j)
      {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}

/* Slightly different loop order to simulate different performance characteristics */
static void kji_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
  if (!A || !B || !C)
    return;
  for (size_t i = 0; i < M * N; ++i)
    C[i] = 0.0f;

  for (size_t k = 0; k < K; ++k)
  {
    for (size_t j = 0; j < N; ++j)
    {
      float b = B[k * N + j];
      for (size_t i = 0; i < M; ++i)
        C[i * N + j] += A[i * K + k] * b;
    }
  }
}

/* Blocked (small 8x8) naive block matmul to simulate a faster kernel for some sizes */
static void blocked_8x8_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
  if (!A || !B || !C)
    return;
  /* Zero C */
  for (size_t i = 0; i < M * N; ++i)
    C[i] = 0.0f;

  const size_t BBS = 8;
  for (size_t ii = 0; ii < M; ii += BBS)
  {
    for (size_t kk = 0; kk < K; kk += BBS)
    {
      for (size_t jj = 0; jj < N; jj += BBS)
      {
        size_t i_max = (ii + BBS > M) ? M : ii + BBS;
        size_t k_max = (kk + BBS > K) ? K : kk + BBS;
        size_t j_max = (jj + BBS > N) ? N : jj + BBS;
        for (size_t i = ii; i < i_max; ++i)
        {
          for (size_t k = kk; k < k_max; ++k)
          {
            float a = A[i * K + k];
            for (size_t j = jj; j < j_max; ++j)
              C[i * N + j] += a * B[k * N + j];
          }
        }
      }
    }
  }
}

/* Exposed symbols matching adaptive_tuner.h called kernels */
void blis_sgemm(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
  blocked_8x8_matmul(A, B, C, M, N, K);
}

void openblas_sgemm(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
  naive_matmul(A, B, C, M, N, K);
}

/* For safety: avoid unused warnings when compiled into some builds */
void __attribute__((unused)) mock_kernels_noop(void)
{
  (void)blis_sgemm;
  (void)openblas_sgemm;
}