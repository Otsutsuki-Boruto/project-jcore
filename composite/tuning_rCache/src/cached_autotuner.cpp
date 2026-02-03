// composite/tuning_cache/src/cached_autotuner.cpp

#include "cached_autotuner.h"
#include "config.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <sys/stat.h>
#include <unistd.h>

#include "pool_manager.h"

// Lightweight cache entry
typedef struct
{
  size_t M, N, K, threads, tile_size;
  char kernel_name[256];
  uint64_t timestamp;
} cache_entry_t;

// Simple hash for operation signature
static uint64_t compute_hash(size_t M, size_t N, size_t K, size_t threads, size_t tile_size)
{
  uint64_t hash = 14695981039346656037ULL;
  hash ^= M;
  hash *= 1099511628211ULL;
  hash ^= N;
  hash *= 1099511628211ULL;
  hash ^= K;
  hash *= 1099511628211ULL;
  hash ^= threads;
  hash *= 1099511628211ULL;
  hash ^= tile_size;
  hash *= 1099511628211ULL;
  return hash;
}

// Hardware signature (simple)
typedef struct
{
  uint32_t cores;
  uint64_t features;
} hw_sig_t;

static hw_sig_t detect_hardware()
{
  hw_sig_t sig = {0};
  sig.cores = static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_ONLN));

  // Simple feature detection
#if defined(__AVX2__)
  sig.features |= (1ULL << 1);
#endif
#if defined(__AVX512F__)
  sig.features |= (1ULL << 2);
#endif

  // Note: More sophisticated detection can use CPUID directly
  // or integrate with jcore_hw_introspect for full topology

  return sig;
}

// Cache structure (simple in-memory hash table)
#define MAX_CACHE_ENTRIES 1024

typedef struct
{
  cache_entry_t entries[MAX_CACHE_ENTRIES];
  int valid[MAX_CACHE_ENTRIES];
  size_t count;
} cache_table_t;

// Main handle structure
struct cat_handle_t
{
  cache_table_t cache;
  cat_config_t config;
  cat_stats_t stats;
  hw_sig_t hw_sig;
  int tuner_initialized;
  char last_kernel[256];
  pm_t *pool_manager;
};

// Ensure directory exists
static int ensure_dir(const char *path)
{
  struct stat st;
  if (stat(path, &st) == 0)
    return (S_ISDIR(st.st_mode)) ? 0 : -1;

  // Try to create
  if (mkdir(path, 0755) == 0)
    return 0;

  // Try creating parent
  char parent[512];
  strncpy(parent, path, sizeof(parent) - 1);
  parent[sizeof(parent) - 1] = '\0';

  char *last_slash = strrchr(parent, '/');
  if (last_slash && last_slash != parent)
  {
    *last_slash = '\0';
    if (ensure_dir(parent) == 0)
    {
      return mkdir(path, 0755) == 0 ? 0 : -1;
    }
  }

  return -1;
}

// Cache lookup
static int cache_lookup(cache_table_t *cache, size_t M, size_t N, size_t K,
                        size_t threads, size_t tile_size, char *out_kernel)
{
  uint64_t hash = compute_hash(M, N, K, threads, tile_size);
  size_t idx = hash % MAX_CACHE_ENTRIES;

  // Linear probing
  for (size_t i = 0; i < MAX_CACHE_ENTRIES; i++)
  {
    size_t pos = (idx + i) % MAX_CACHE_ENTRIES;
    if (!cache->valid[pos])
      continue;

    cache_entry_t *e = &cache->entries[pos];
    if (e->M == M && e->N == N && e->K == K &&
        e->threads == threads && e->tile_size == tile_size)
    {
      strcpy(out_kernel, e->kernel_name);
      return 1; // Hit
    }
  }
  return 0; // Miss
}

// Cache insert
static void cache_insert(cache_table_t *cache, size_t M, size_t N, size_t K,
                         size_t threads, size_t tile_size, const char *kernel)
{
  uint64_t hash = compute_hash(M, N, K, threads, tile_size);
  size_t idx = hash % MAX_CACHE_ENTRIES;

  // Find empty slot or overwrite oldest
  size_t pos = idx;
  uint64_t oldest_ts = UINT64_MAX;
  size_t oldest_pos = idx;

  for (size_t i = 0; i < MAX_CACHE_ENTRIES; i++)
  {
    size_t p = (idx + i) % MAX_CACHE_ENTRIES;
    if (!cache->valid[p])
    {
      pos = p;
      break;
    }
    if (cache->entries[p].timestamp < oldest_ts)
    {
      oldest_ts = cache->entries[p].timestamp;
      oldest_pos = p;
    }
  }

  if (cache->valid[pos])
    pos = oldest_pos; // Evict oldest

  cache_entry_t *e = &cache->entries[pos];
  e->M = M;
  e->N = N;
  e->K = K;
  e->threads = threads;
  e->tile_size = tile_size;
  strncpy(e->kernel_name, kernel, sizeof(e->kernel_name) - 1);
  e->kernel_name[sizeof(e->kernel_name) - 1] = '\0';
  e->timestamp = static_cast<uint64_t>(time(nullptr));
  cache->valid[pos] = 1;

  if (!cache->valid[pos] || pos == oldest_pos)
    cache->count++;
}

// Save cache (binary format)
static int save_cache_binary(cat_handle_t *h, const char *filepath)
{
  FILE *f = fopen(filepath, "wb");
  if (!f)
    return -1;

  // Write header
  uint32_t magic = 0x4A434348; // "JCCH"
  uint32_t version = 1;
  fwrite(&magic, sizeof(magic), 1, f);
  fwrite(&version, sizeof(version), 1, f);

  // Write hardware signature
  fwrite(&h->hw_sig, sizeof(h->hw_sig), 1, f);

  // Write count
  uint32_t count = 0;
  for (size_t i = 0; i < MAX_CACHE_ENTRIES; i++)
  {
    if (h->cache.valid[i])
      count++;
  }
  fwrite(&count, sizeof(count), 1, f);

  // Write entries
  for (size_t i = 0; i < MAX_CACHE_ENTRIES; i++)
  {
    if (h->cache.valid[i])
    {
      fwrite(&h->cache.entries[i], sizeof(cache_entry_t), 1, f);
    }
  }

  fclose(f);
  return 0;
}

// Load cache (binary format)
static int load_cache_binary(cat_handle_t *h, const char *filepath)
{
  FILE *f = fopen(filepath, "rb");
  if (!f)
    return -1;

  // Read header
  uint32_t magic = 0, version = 0;
  if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x4A434348)
  {
    fclose(f);
    return -1;
  }
  if (fread(&version, sizeof(version), 1, f) != 1 || version != 1)
  {
    fclose(f);
    return -1;
  }

  // Read hardware signature
  hw_sig_t file_hw;
  if (fread(&file_hw, sizeof(file_hw), 1, f) != 1)
  {
    fclose(f);
    return -1;
  }

  // Validate hardware if needed
  if (h->config.validate_hardware)
  {
    if (file_hw.cores != h->hw_sig.cores || file_hw.features != h->hw_sig.features)
    {
      fclose(f);
      return -1; // Hardware mismatch
    }
  }

  // Read count
  uint32_t count = 0;
  if (fread(&count, sizeof(count), 1, f) != 1)
  {
    fclose(f);
    return -1;
  }

  // Clear existing cache
  memset(h->cache.valid, 0, sizeof(h->cache.valid));
  h->cache.count = 0;

  // Read entries
  for (uint32_t i = 0; i < count && i < MAX_CACHE_ENTRIES; i++)
  {
    cache_entry_t e;
    if (fread(&e, sizeof(e), 1, f) != 1)
      break;

    // Insert into cache
    cache_insert(&h->cache, e.M, e.N, e.K, e.threads, e.tile_size, e.kernel_name);
  }

  fclose(f);
  return 0;
}

// Get cache file path
static void get_cache_path(cat_handle_t *h, char *out, size_t len)
{
  snprintf(out, len, "%s/tuning_cache.bin", h->config.cache_dir);
}

// API Implementation

void cat_config_init_default(cat_config_t *config)
{
  if (!config)
    return;

  // Default cache directory
  const char *home = getenv("HOME");
  if (home)
    snprintf(config->cache_dir, sizeof(config->cache_dir), "%s/.jcore/cache", home);
  else
    strncpy(config->cache_dir, "/tmp/jcore_cache", sizeof(config->cache_dir) - 1);

  // Read environment variables
  const char *env_dir = getenv("JCORE_CACHE_DIR");
  if (env_dir && env_dir[0])
    strncpy(config->cache_dir, env_dir, sizeof(config->cache_dir) - 1);

  config->format = CAT_FORMAT_BINARY;
  config->max_entries = 0; // Unlimited
  config->validate_hardware = 1;
  config->auto_save = 1;
  config->force_benchmark = 0;

  const char *env_fmt = getenv("JCORE_CACHE_FORMAT");
  if (env_fmt && strcmp(env_fmt, "json") == 0)
    config->format = CAT_FORMAT_JSON;

  const char *env_validate = getenv("JCORE_CACHE_VALIDATE_HW");
  if (env_validate && strcmp(env_validate, "0") == 0)
    config->validate_hardware = 0;

  const char *env_auto = getenv("JCORE_CACHE_AUTO_SAVE");
  if (env_auto && strcmp(env_auto, "0") == 0)
    config->auto_save = 0;
}

cat_status_t cat_init(cat_handle_t **out_handle)
{
  cat_config_t config;
  cat_config_init_default(&config);
  return cat_init_with_config(out_handle, &config);
}

cat_status_t cat_init_with_config(cat_handle_t **out_handle, const cat_config_t *config) {
  if (!out_handle || !config)
    return CAT_ERR_INVALID_ARG;

  cat_handle_t *h = static_cast<cat_handle_t *>(calloc(1, sizeof(cat_handle_t)));
  if (!h)
    return CAT_ERR_NO_MEMORY;

  h->config = *config;
  h->hw_sig = detect_hardware();
  // TRCS does NOT own autotuner lifecycle or kernel registration.
  // Kernel dispatch is the single source of truth.
  h->tuner_initialized = 0;
  h->pool_manager = nullptr;

  // Initialize memory pool manager for cache operations
  size_t pool_size = 32 * 1024 * 1024; // 32 MB
  size_t chunk_size = sizeof(cache_entry_t); // Size of one cache entry
  pm_status_t pm_status = pm_init(&h->pool_manager, pool_size, chunk_size, 0, -1);
  if (pm_status != PM_OK)
  {
    h->pool_manager = nullptr;
  }


  // Ensure cache directory exists
  if (ensure_dir(h->config.cache_dir) != 0) {
    // Autotuner lifecycle is owned by kernel dispatch
    free(h);
    return CAT_ERR_CACHE_FAILURE;
  }

  // Try to load existing cache
  char cache_path[600];
  get_cache_path(h, cache_path, sizeof(cache_path));
  load_cache_binary(h, cache_path);

  *out_handle = h;
  return CAT_OK;
}

void cat_shutdown(cat_handle_t *handle) {
  if (!handle)
    return;

  // Auto-save if enabled
  if (handle->config.auto_save) {
    char cache_path[600];
    get_cache_path(handle, cache_path, sizeof(cache_path));
    save_cache_binary(handle, cache_path);
  }

  // Only shutdown tuner if we initialized it
  if (handle->tuner_initialized) {
    at_shutdown();
  }

  // Shutdown pool manager
  if (handle->pool_manager) {
    pm_shutdown(handle->pool_manager);
    handle->pool_manager = nullptr;
  }

  free(handle);
}

const char *cat_select_kernel(cat_handle_t *handle,
                              size_t M, size_t N, size_t K,
                              size_t threads, size_t tile_size)
{
  if (!handle)
    return nullptr;

  // Check cache first (unless force benchmark)
  if (!handle->config.force_benchmark)
  {
    if (cache_lookup(&handle->cache, M, N, K, threads, tile_size, handle->last_kernel))
    {
      handle->stats.cache_hits++;
      return handle->last_kernel;
    }
  }

  handle->stats.cache_misses++;

  // Run benchmark
  at_status_t at_s = at_benchmark_matmul_all(M, N, K, threads, tile_size,
                                             handle->last_kernel, sizeof(handle->last_kernel));
  if (at_s != AT_OK)
  {
    // Provide fallback for any benchmark failure
    const char* fallback = (at_s == AT_ERR_NO_KERNELS) ? "no_kernels" :
                           (at_s == AT_ERR_BENCHMARK_FAIL) ? "openblas" : "default";

    strncpy(handle->last_kernel, fallback, sizeof(handle->last_kernel) - 1);
    handle->last_kernel[sizeof(handle->last_kernel) - 1] = '\0';

    // Still continue - cache will store the fallback decision
  }

  handle->stats.benchmarks_run++;

  // Store in cache
  cache_insert(&handle->cache, M, N, K, threads, tile_size, handle->last_kernel);
  handle->stats.total_entries = handle->cache.count;

  // Auto-save if enabled
  if (handle->config.auto_save)
  {
    char cache_path[600];
    get_cache_path(handle, cache_path, sizeof(cache_path));
    save_cache_binary(handle, cache_path);
  }

  return handle->last_kernel;
}

cat_status_t cat_force_benchmark(cat_handle_t *handle,
                                 size_t M, size_t N, size_t K,
                                 size_t threads, size_t tile_size,
                                 char *out_kernel, size_t out_len)
{
  if (!handle || !out_kernel || out_len == 0)
    return CAT_ERR_INVALID_ARG;

  char kernel[256];
  at_status_t at_s = at_benchmark_matmul_all(M, N, K, threads, tile_size,
                                             kernel, sizeof(kernel));
  if (at_s != AT_OK)
  {
    // Provide fallback for benchmark failure
    const char* fallback = (at_s == AT_ERR_NO_KERNELS) ? "no_kernels" :
                           (at_s == AT_ERR_BENCHMARK_FAIL) ? "openblas" : "default";

    strncpy(kernel, fallback, sizeof(kernel) - 1);
    kernel[sizeof(kernel) - 1] = '\0';
  }

  handle->stats.benchmarks_run++;

  // Update cache
  cache_insert(&handle->cache, M, N, K, threads, tile_size, kernel);
  handle->stats.total_entries = handle->cache.count;

  strncpy(out_kernel, kernel, out_len - 1);
  out_kernel[out_len - 1] = '\0';

  return CAT_OK;
}

cat_status_t cat_get_stats(cat_handle_t *handle, cat_stats_t *out_stats)
{
  if (!handle || !out_stats)
    return CAT_ERR_INVALID_ARG;

  *out_stats = handle->stats;
  size_t total = handle->stats.cache_hits + handle->stats.cache_misses;
  if (total > 0)
    out_stats->hit_rate = static_cast<double>(handle->stats.cache_hits) / total;
  else
    out_stats->hit_rate = 0.0;

  return CAT_OK;
}

cat_status_t cat_clear_cache(cat_handle_t *handle)
{
  if (!handle)
    return CAT_ERR_INVALID_ARG;

  memset(handle->cache.valid, 0, sizeof(handle->cache.valid));
  handle->cache.count = 0;
  handle->stats.total_entries = 0;

  return CAT_OK;
}

cat_status_t cat_save_cache(cat_handle_t *handle)
{
  if (!handle)
    return CAT_ERR_INVALID_ARG;

  char cache_path[600];
  get_cache_path(handle, cache_path, sizeof(cache_path));

  if (save_cache_binary(handle, cache_path) != 0)
    return CAT_ERR_CACHE_FAILURE;

  return CAT_OK;
}

cat_status_t cat_load_cache(cat_handle_t *handle)
{
  if (!handle)
    return CAT_ERR_INVALID_ARG;

  char cache_path[600];
  get_cache_path(handle, cache_path, sizeof(cache_path));

  if (load_cache_binary(handle, cache_path) != 0)
    return CAT_ERR_CACHE_FAILURE;

  return CAT_OK;
}

cat_status_t cat_export_cache(cat_handle_t *handle, const char *filepath, cat_format_t format)
{
  if (!handle || !filepath)
    return CAT_ERR_INVALID_ARG;

  if (format == CAT_FORMAT_BINARY)
  {
    if (save_cache_binary(handle, filepath) != 0)
      return CAT_ERR_CACHE_FAILURE;
  }
  else
  {
    // JSON not implemented in lightweight version
    return CAT_ERR_INTERNAL;
  }

  return CAT_OK;
}

cat_status_t cat_import_cache(cat_handle_t *handle, const char *filepath)
{
  if (!handle || !filepath)
    return CAT_ERR_INVALID_ARG;

  if (load_cache_binary(handle, filepath) != 0)
    return CAT_ERR_CACHE_FAILURE;

  return CAT_OK;
}

const char *cat_status_str(cat_status_t s)
{
  switch (s)
  {
  case CAT_OK:
    return "OK";
  case CAT_ERR_NOT_INITIALIZED:
    return "Not initialized";
  case CAT_ERR_INVALID_ARG:
    return "Invalid argument";
  case CAT_ERR_NO_MEMORY:
    return "Out of memory";
  case CAT_ERR_CACHE_FAILURE:
    return "Cache operation failed";
  case CAT_ERR_TUNER_FAILURE:
    return "Tuner operation failed";
  case CAT_ERR_INTERNAL:
    return "Internal error";
  default:
    return "Unknown error";
  }
}