#ifndef KERNELS_OPENBLAS_H_
#define KERNELS_OPENBLAS_H_
#include "adaptive_tuner.h"

#ifdef __cplusplus
extern "C"
{
#endif

  void openblas_sgemm(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

#ifdef __cplusplus
}
#endif

#endif
// KERNELS_OPENBLAS_H_