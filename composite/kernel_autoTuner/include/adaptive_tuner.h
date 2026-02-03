#ifndef ADAPTIVE_TUNER_H_
#define ADAPTIVE_TUNER_H_

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum
  {
    AT_OK = 0,
    AT_ERR_NO_MEMORY,
    AT_ERR_INVALID_ARG,
    AT_ERR_NOT_INITIALIZED,
    AT_ERR_NO_KERNELS,
    AT_ERR_BENCHMARK_FAIL,
    AT_ERR_CONFLICT,
    AT_ERR_INTERNAL
  } at_status_t;

  // FFM-compatible matmul function pointer
  typedef void (*jcore_matmul_f32_fn)(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

  // FFM API
  at_status_t at_init(void);
  void at_shutdown(void);
  at_status_t at_register_matmul_impl(const char *name, unsigned long long required_features,
                                      jcore_matmul_f32_fn fn);
  at_status_t at_benchmark_matmul_all(size_t M, size_t N, size_t K,
                                      size_t preferred_threads, size_t tile_size_hint,
                                      char *best_name, size_t best_name_len);
  const char *at_status_str(at_status_t s);

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_TUNER_H_
