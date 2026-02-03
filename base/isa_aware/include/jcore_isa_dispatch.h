#ifndef JCORE_ISA_DISPATCH_H
#define JCORE_ISA_DISPATCH_H

#ifdef _WIN32
#ifdef JCORE_BUILD_DLL
#define JCORE_API __declspec(dllexport)
#else
#define JCORE_API __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define JCORE_API __attribute__((visibility("default")))
#else
#define JCORE_API
#endif
#endif

#include <stdint.h>
#include <stddef.h> /* size_t */

/* --- Feature bitmask --- */
typedef uint64_t jcore_features_t;

enum
{
  JCORE_FEAT_NONE = 0ULL,
  JCORE_FEAT_SSE2 = 1ULL << 0,
  JCORE_FEAT_AVX = 1ULL << 1,
  JCORE_FEAT_AVX2 = 1ULL << 2,
  JCORE_FEAT_AVX512 = 1ULL << 3,
  JCORE_FEAT_AMX = 1ULL << 4,
};

/* --- Error codes --- */
typedef enum
{
  JCORE_OK = 0,
  JCORE_ERR_NO_MEMORY = -1,
  JCORE_ERR_INVALID_ARG = -2,
  JCORE_ERR_NOT_FOUND = -3,
  JCORE_ERR_CONFLICT = -4,
  JCORE_ERR_INTERNAL = -5,
  JCORE_ERR_NOT_INITIALIZED = -6,
  JCORE_ERR_REG_LIMIT = -7,
} jcore_err_t;

/* --- Operation identifiers --- */
typedef uint32_t jcore_op_t;
enum
{
  JCORE_OP_VECTOR_ADD = 1,
  JCORE_OP_MATMUL = 2,
};

/* --- Function pointer typedefs --- */
typedef void (*jcore_vec_add_f32_fn)(const float *, const float *, float *, size_t);
typedef void (*jcore_matmul_f32_fn)(const float *, const float *, float *, size_t, size_t, size_t);
typedef void (*jcore_generic_fn)(void *);

/* --- Argument structs for generic dispatch (exposed to callers/FFI) --- */
/* These structs must match the ones used by implementations. */
typedef struct
{
  const float *a;
  const float *b;
  float *dest;
  size_t len;
} jcore_vec_add_args_t;

typedef struct
{
  const float *A;
  const float *B;
  float *C;
  size_t M;
  size_t N;
  size_t K;
} jcore_matmul_args_t;

/* --- Public API (FFM/ABI-safe C exports) --- */
#ifdef __cplusplus
extern "C"
{
#endif

  JCORE_API int jcore_init_dispatch(void);
  JCORE_API jcore_features_t jcore_get_host_features(void);

  /* Register implementation for an op. required_feats = bitmask of features needed. */
  JCORE_API int jcore_register_impl(jcore_op_t op, jcore_features_t required_feats, jcore_generic_fn fn);

  /* Generic dispatch: caller supplies pointer to op's arg struct (see above). */
  JCORE_API int jcore_dispatch(jcore_op_t op, void *args);

  /* Typed convenience wrappers (callers may use these or call jcore_dispatch directly). */
  JCORE_API int jcore_dispatch_vec_add_f32(const float *a, const float *b, float *dest, size_t len);
  JCORE_API int jcore_dispatch_matmul_f32(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

  /* Register built-in scalar fallbacks (convenience). */
  JCORE_API int jcore_register_scalar_fallbacks(void);

  /* Convert an error code to a readable message (helpful for debugging). */
  JCORE_API const char *jcore_strerror(int err);

  /* Add this declaration for the scalar fallback implementation */
  JCORE_API void jcore_scalar_matmul_impl(void *vargs);

#ifdef __cplusplus
}
#endif

#endif /* JCORE_ISA_DISPATCH_H */
