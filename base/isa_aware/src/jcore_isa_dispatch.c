/* Dispatch core: registration, best-match selection, dispatch calling,
 * typed wrappers, and error strings.
 */

#include "jcore_isa_dispatch.h"
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

extern jcore_features_t jcore_detect_cpu_features_x86(void);

#define JCORE_MAX_IMPLS 128u

struct jcore_impl_desc_t
{
  jcore_op_t op;
  jcore_features_t required;
  jcore_generic_fn fn;
};

static struct
{
  pthread_mutex_t lock;
  _Atomic int initialized;
  jcore_features_t host_features;
  struct jcore_impl_desc_t impls[JCORE_MAX_IMPLS];
  size_t impl_count;
} g_dispatch = {PTHREAD_MUTEX_INITIALIZER, 0, JCORE_FEAT_NONE, {0}, 0};

/* Internal one-time features init */
static int jcore_init_features_once_internal(void)
{
  int expected = 0;
  if (atomic_compare_exchange_strong(&g_dispatch.initialized, &expected, 1))
  {
    jcore_features_t f = jcore_detect_cpu_features_x86();
    atomic_store(&g_dispatch.host_features, f);
  }
  else
  {
    while (atomic_load(&g_dispatch.initialized) != 1)
    { /* spin until ready */
      ;
    }
  }
  return 0;
}

int jcore_init_dispatch(void)
{
  jcore_init_features_once_internal();
  return JCORE_OK;
}

jcore_features_t jcore_get_host_features(void)
{
  if (!atomic_load(&g_dispatch.initialized))
    jcore_init_features_once_internal();
  return atomic_load(&g_dispatch.host_features);
}

/* score: -1 if requires features not present, else number of required bits (prefer more specialized). */
static int jcore_score_impl(const struct jcore_impl_desc_t *impl, jcore_features_t host)
{
  if ((impl->required & ~host) != 0)
    return -1;
  jcore_features_t m = impl->required;
  int score = 0;
  while (m)
  {
    score += (m & 1) ? 1 : 0;
    m >>= 1;
  }
  return score;
}

static int jcore_find_best_impl_locked(jcore_op_t op, jcore_features_t host)
{
  int best_idx = -1;
  int best_score = -1;
  for (size_t i = 0; i < g_dispatch.impl_count; ++i)
  {
    const struct jcore_impl_desc_t *d = &g_dispatch.impls[i];
    if (d->op != op)
      continue;
    int s = jcore_score_impl(d, host);
    if (s < 0)
      continue;
    if (s > best_score)
    {
      best_score = s;
      best_idx = (int)i;
    }
  }
  return best_idx;
}

/* Register implementation */
int jcore_register_impl(jcore_op_t op, jcore_features_t required_feats, jcore_generic_fn fn)
{
  if (!fn)
    return JCORE_ERR_INVALID_ARG;
  if (!atomic_load(&g_dispatch.initialized))
    jcore_init_features_once_internal();
  if (pthread_mutex_lock(&g_dispatch.lock) != 0)
    return JCORE_ERR_INTERNAL;
  if (g_dispatch.impl_count >= JCORE_MAX_IMPLS)
  {
    pthread_mutex_unlock(&g_dispatch.lock);
    return JCORE_ERR_REG_LIMIT;
  }
  struct jcore_impl_desc_t *desc = &g_dispatch.impls[g_dispatch.impl_count++];
  desc->op = op;
  desc->required = required_feats;
  desc->fn = fn;
  pthread_mutex_unlock(&g_dispatch.lock);
  return JCORE_OK;
}

/* Generic dispatch: finds best registered impl and invokes it (outside lock). */
int jcore_dispatch(jcore_op_t op, void *args)
{
  if (!atomic_load(&g_dispatch.initialized))
    jcore_init_features_once_internal();

  if (pthread_mutex_lock(&g_dispatch.lock) != 0)
    return JCORE_ERR_INTERNAL;
  if (g_dispatch.impl_count == 0)
  {
    pthread_mutex_unlock(&g_dispatch.lock);
    return JCORE_ERR_NOT_FOUND;
  }

  jcore_features_t host = atomic_load(&g_dispatch.host_features);
  int idx = jcore_find_best_impl_locked(op, host);
  if (idx < 0)
  {
    pthread_mutex_unlock(&g_dispatch.lock);
    return JCORE_ERR_NOT_FOUND;
  }

  jcore_generic_fn fn = g_dispatch.impls[idx].fn;
  pthread_mutex_unlock(&g_dispatch.lock);

  if (!fn)
    return JCORE_ERR_INTERNAL;
  fn(args);
  return JCORE_OK;
}

/* Typed convenience wrappers that construct arg structs and call jcore_dispatch(). */
int jcore_dispatch_vec_add_f32(const float *a, const float *b, float *dest, size_t len)
{
  if (!a || !b || !dest)
    return JCORE_ERR_INVALID_ARG;
  jcore_vec_add_args_t args = {.a = a, .b = b, .dest = dest, .len = len};
  return jcore_dispatch(JCORE_OP_VECTOR_ADD, &args);
}

int jcore_dispatch_matmul_f32(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
  if (!A || !B || !C)
    return JCORE_ERR_INVALID_ARG;
  jcore_matmul_args_t args = {.A = A, .B = B, .C = C, .M = M, .N = N, .K = K};
  return jcore_dispatch(JCORE_OP_MATMUL, &args);
}

/* Error -> readable message (helpful for logging / debugging) */
const char *jcore_strerror(int err)
{
  switch (err)
  {
  case JCORE_OK:
    return "JCORE_OK";
  case JCORE_ERR_NO_MEMORY:
    return "JCORE_ERR_NO_MEMORY";
  case JCORE_ERR_INVALID_ARG:
    return "JCORE_ERR_INVALID_ARG";
  case JCORE_ERR_NOT_FOUND:
    return "JCORE_ERR_NOT_FOUND";
  case JCORE_ERR_CONFLICT:
    return "JCORE_ERR_CONFLICT";
  case JCORE_ERR_INTERNAL:
    return "JCORE_ERR_INTERNAL";
  case JCORE_ERR_NOT_INITIALIZED:
    return "JCORE_ERR_NOT_INITIALIZED";
  case JCORE_ERR_REG_LIMIT:
    return "JCORE_ERR_REG_LIMIT";
  default:
    return "JCORE_ERR_UNKNOWN";
  }
}
