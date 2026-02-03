#ifndef KERNELS_BLIS_H_
#define KERNELS_BLIS_H_

#include "adaptive_tuner.h"

#ifdef __cplusplus
extern "C"
{
#endif

  void blis_sgemm(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

#ifdef __cplusplus
}
#endif

#endif
// KERNELS_BLIS_H_