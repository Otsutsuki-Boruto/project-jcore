#include "mem_wrapper.h"
#include "numa_helper.h"

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef USE_JEMALLOC
// Ensure jemalloc exposes the je_* symbols properly
#define JEMALLOC_NO_DEMANGLE 1
#include <jemalloc/jemalloc.h>
#endif

#ifdef USE_MEMKIND
#include <memkind.h>
#endif

/* -------------------------------------------------------------------------- */
/*                        Global Backend Configuration                        */
/* -------------------------------------------------------------------------- */

static ffm_backend_t g_backend = FFM_BACKEND_MALLOC;
static int g_initialized = 0;

/* -------------------------------------------------------------------------- */
/*                        Internal Helper Functions                           */
/* -------------------------------------------------------------------------- */

static void *ffm_alloc_internal(size_t size, size_t alignment)
{
  void *p = NULL;

  switch (g_backend)
  {
  case FFM_BACKEND_JEMALLOC:
#ifdef USE_JEMALLOC
    if (alignment)
    {
      if (je_posix_memalign(&p, alignment, size) != 0)
        p = NULL;
    }
    else
    {
      p = je_malloc(size);
    }
#else
    if (alignment)
      posix_memalign(&p, alignment, size);
    else
      p = malloc(size);
#endif
    break;

  case FFM_BACKEND_MEMKIND:
#ifdef USE_MEMKIND
    p = memkind_malloc(MEMKIND_DEFAULT, size);
#else
    if (alignment)
      posix_memalign(&p, alignment, size);
    else
      p = malloc(size);
#endif
    break;

  case FFM_BACKEND_MALLOC:
  default:
    if (alignment)
      posix_memalign(&p, alignment, size);
    else
      p = malloc(size);
    break;
  }

  if (!p)
    errno = ENOMEM;

  return p;
}

/* -------------------------------------------------------------------------- */
/*                           Public API Implementation                        */
/* -------------------------------------------------------------------------- */

ffm_status_t ffm_init(ffm_backend_t backend)
{
  if (g_initialized)
    return FFM_OK;

  if (backend == FFM_BACKEND_AUTO)
  {
#ifdef USE_JEMALLOC
    g_backend = FFM_BACKEND_JEMALLOC;
#elif defined(USE_MEMKIND)
    g_backend = FFM_BACKEND_MEMKIND;
#else
    g_backend = FFM_BACKEND_MALLOC;
#endif
  }
  else
  {
    g_backend = backend;
  }

  g_initialized = 1;
  return FFM_OK;
}

void ffm_shutdown(void)
{
  g_initialized = 0;
  g_backend = FFM_BACKEND_MALLOC;
}

void *ffm_malloc(size_t size)
{
  return ffm_alloc_internal(size, 0);
}

void *ffm_calloc(size_t nmemb, size_t size)
{
  size_t total = nmemb * size;
  void *ptr = ffm_malloc(total);
  if (ptr)
    memset(ptr, 0, total);
  return ptr;
}

void *ffm_aligned_alloc(size_t alignment, size_t size)
{
  return ffm_alloc_internal(size, alignment);
}

void ffm_free(void *ptr)
{
  if (!ptr)
    return;

  switch (g_backend)
  {
  case FFM_BACKEND_JEMALLOC:
#ifdef USE_JEMALLOC
    je_free(ptr);
#else
    free(ptr);
#endif
    break;

  case FFM_BACKEND_MEMKIND:
#ifdef USE_MEMKIND
    memkind_free(MEMKIND_DEFAULT, ptr);
#else
    free(ptr);
#endif
    break;

  case FFM_BACKEND_MALLOC:
  default:
    free(ptr);
    break;
  }
}

void *ffm_realloc(void *ptr, size_t new_size)
{
  if (!g_initialized)
  {
    errno = EINVAL;
    return NULL;
  }

  if (new_size == 0)
    new_size = 1;

  void *p = NULL;

  switch (g_backend)
  {
  case FFM_BACKEND_JEMALLOC:
#ifdef USE_JEMALLOC
    p = je_realloc(ptr, new_size);
#else
    p = realloc(ptr, new_size);
#endif
    break;

  case FFM_BACKEND_MEMKIND:
#ifdef USE_MEMKIND
    p = memkind_realloc(MEMKIND_DEFAULT, ptr, new_size);
#else
    p = realloc(ptr, new_size);
#endif
    break;

  case FFM_BACKEND_MALLOC:
  default:
    p = realloc(ptr, new_size);
    break;
  }

  if (!p)
    errno = ENOMEM;

  return p;
}

ffm_status_t ffm_set_numa_node(int node)
{
  int rc = numa_helper_set_node(node);
  return (rc == 0) ? FFM_OK : FFM_ERR_UNSUPPORTED;
}

ffm_backend_t ffm_get_backend(void)
{
  return g_backend;
}

const char *ffm_status_str(ffm_status_t s)
{
  switch (s)
  {
  case FFM_OK:
    return "FFM_OK";
  case FFM_ERR_NO_MEMORY:
    return "FFM_ERR_NO_MEMORY";
  case FFM_ERR_INVALID_ARG:
    return "FFM_ERR_INVALID_ARG";
  case FFM_ERR_NOT_INITIALIZED:
    return "FFM_ERR_NOT_INITIALIZED";
  case FFM_ERR_UNSUPPORTED:
    return "FFM_ERR_UNSUPPORTED";
  case FFM_ERR_INTERNAL:
    return "FFM_ERR_INTERNAL";
  case FFM_ERR_INIT:
    return "FFM_ERR_INIT";
  default:
    return "FFM_ERR_UNKNOWN";
  }
}
