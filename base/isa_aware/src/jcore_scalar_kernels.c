/* Scalar fallback kernels (use the public arg structs from the header) */

#include "jcore_isa_dispatch.h"

static void jcore_scalar_vec_add_impl(void *vargs)
{
  jcore_vec_add_args_t *args = (jcore_vec_add_args_t *)vargs;
  const float *a = args->a;
  const float *b = args->b;
  float *d = args->dest;
  size_t n = args->len;
  for (size_t i = 0; i < n; ++i)
    d[i] = a[i] + b[i];
}

void jcore_scalar_matmul_impl(void *vargs)
{
  jcore_matmul_args_t *args = (jcore_matmul_args_t *)vargs;
  const float *A = args->A;
  const float *B = args->B;
  float *C = args->C;
  size_t M = args->M, N = args->N, K = args->K;
  for (size_t i = 0; i < M; ++i)
  {
    for (size_t j = 0; j < N; ++j)
    {
      float s = 0.0f;
      for (size_t k = 0; k < K; ++k)
        s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
  }
}

int jcore_register_scalar_fallbacks(void)
{
  int rc = jcore_register_impl(JCORE_OP_VECTOR_ADD, JCORE_FEAT_NONE, (jcore_generic_fn)jcore_scalar_vec_add_impl);
  if (rc != JCORE_OK)
    return rc;
  rc = jcore_register_impl(JCORE_OP_MATMUL, JCORE_FEAT_NONE, (jcore_generic_fn)jcore_scalar_matmul_impl);
  return rc;
}
