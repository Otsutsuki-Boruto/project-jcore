#include "jcore_isa_dispatch.h"
#include "adaptive_tuner.h"
#include "cpu_features.h"
#include <unordered_map>
#include <mutex>

static std::unordered_map<jcore_op_t, std::pair<jcore_features_t, jcore_generic_fn>> registry;
static jcore_features_t host_features = 0;
static std::mutex reg_mutex;

extern "C"
{
  // Forward declarations to real scalar fallback implementations
  extern void jcore_scalar_matmul_impl(void *vargs);
  extern void jcore_scalar_vec_add_impl(void *vargs);
  extern int jcore_register_scalar_fallbacks(void); // base component function

  int jcore_init_dispatch(void)
  {
    CPUFeatures f = detect_cpu_features();
    host_features = 0;
    if (f.avx)
      host_features |= JCORE_FEAT_AVX;
    if (f.avx2)
      host_features |= JCORE_FEAT_AVX2;
    if (f.avx512)
      host_features |= JCORE_FEAT_AVX512;
    if (f.amx)
      host_features |= JCORE_FEAT_AMX;
    return JCORE_OK;
  }

  jcore_features_t jcore_get_host_features(void) { return host_features; }

  int jcore_register_impl(jcore_op_t op, jcore_features_t feats, jcore_generic_fn fn)
  {
    if (!fn)
      return JCORE_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(reg_mutex);
    registry[op] = {feats, fn};
    return JCORE_OK;
  }

  extern "C" int jcore_register_scalar_fallbacks(void); // declare base

  const char *jcore_strerror(int err) { return "Error string TBD"; }

} // extern "C"