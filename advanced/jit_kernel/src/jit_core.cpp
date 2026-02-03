// advanced/jit_kernel/src/jit_core.cpp
#include "jit_kernel_internal.h"
#include <cstdarg>
#include <cstdio>

#include "jcore_hw_introspect.h"
#include "jcore_isa_dispatch.h"

namespace jkg_internal {

// Global state definition
JKGState g_jkg_state;

/* ========================================================================== */
/* Logging Utilities */
/* ========================================================================== */

void log_error(const char *format, ...) {
  if (g_jkg_state.config.verbose) {
    fprintf(stderr, "[JKG ERROR] ");
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n");
  }
}

void log_info(const char *format, ...) {
  if (g_jkg_state.config.verbose) {
    printf("[JKG INFO] ");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
  }
}

/* ========================================================================== */
/* ISA Detection and Conversion */
/* ========================================================================== */

uint32_t detect_available_isa() {
  uint32_t mask = JKG_ISA_SCALAR; // Always available

  CPUFeatures features = detect_cpu_features();

  if (features.avx)
    mask |= JKG_ISA_AVX;
  if (features.avx2)
    mask |= JKG_ISA_AVX2;
  if (features.avx512) {
    mask |= JKG_ISA_AVX512F | JKG_ISA_AVX512BW;
  }
  // AMX requires AVX-512 foundation
  if (features.amx && features.avx512)
    mask |= JKG_ISA_AMX;

  return mask;
}

jkg_isa_t convert_cpu_features_to_isa(const CPUFeatures &features) {
  if (features.avx512)
    return JKG_ISA_AVX512F;
  if (features.avx2)
    return JKG_ISA_AVX2;
  if (features.avx)
    return JKG_ISA_AVX;
  return JKG_ISA_SSE2; // Baseline for x86_64
}

jkg_isa_t select_best_isa(uint32_t available_mask) {
  // Select highest available ISA
  if (available_mask & JKG_ISA_AVX512F)
    return JKG_ISA_AVX512F;
  if (available_mask & JKG_ISA_AVX2)
    return JKG_ISA_AVX2;
  if (available_mask & JKG_ISA_AVX)
    return JKG_ISA_AVX;
  if (available_mask & JKG_ISA_SSE2)
    return JKG_ISA_SSE2;
  return JKG_ISA_SCALAR;
}

/* ========================================================================== */
/* Tile Size Optimization */
/* ========================================================================== */

void compute_optimal_tile_sizes(jkg_isa_t isa,
                                const std::vector<CacheInfo> &cache,
                                size_t &out_M, size_t &out_N, size_t &out_K) {
  // Find L1 and L2 cache sizes
  int l1_kb = 32;  // Default
  int l2_kb = 256; // Default

  for (const auto &c : cache) {
    if (c.level == "0" || c.level == "L1") {
      if (c.type == "Data" || c.type == "Unified") {
        l1_kb = c.size_kb;
      }
    } else if (c.level == "2" || c.level == "L2") {
      l2_kb = c.size_kb;
    }
  }

  // Vector width in floats
  size_t vec_width = 4; // SSE baseline
  if (isa == JKG_ISA_AVX || isa == JKG_ISA_AVX2)
    vec_width = 8;
  if (isa == JKG_ISA_AVX512F)
    vec_width = 16;

  // Compute tile sizes targeting L1 cache
  // Goal: A_tile + B_tile + C_tile fits in L1
  // A: M x K, B: K x N, C: M x N
  // Total: M*K + K*N + M*N floats * 4 bytes

  size_t l1_bytes = l1_kb * 1024;
  size_t l1_floats = l1_bytes / 4;

  // Heuristic: N should be multiple of vector width for efficient vectorization
  out_N = vec_width * 6; // 24, 48, or 96 depending on ISA

  // M should allow register blocking (typically 4-8 rows)
  out_M = 6;

  // K should maximize L1 usage
  // Solve: M*K + K*N + M*N = l1_floats
  // K = (l1_floats - M*N) / (M + N)
  size_t k_estimate = (l1_floats - out_M * out_N) / (out_M + out_N);
  out_K = std::max<size_t>(vec_width, std::min<size_t>(256, k_estimate));
  out_K = (out_K / vec_width) * vec_width; // Align to vector width

  log_info(
      "Optimal tile sizes: M=%zu, N=%zu, K=%zu (L1=%dKB, ISA vec_width=%zu)",
      out_M, out_N, out_K, l1_kb, vec_width);
}

/* ========================================================================== */
/* Cache Management */
/* ========================================================================== */

std::shared_ptr<jkg_kernel_impl_t>
lookup_cached_kernel(const KernelCacheKey &key) {
  std::lock_guard<std::mutex> lock(g_jkg_state.cache_mutex);

  auto it = g_jkg_state.kernel_cache.find(key);
  if (it != g_jkg_state.kernel_cache.end()) {
    g_jkg_state.cache_hits++;
    log_info("Cache hit for kernel (type=%d, M=%zu, N=%zu, K=%zu)", key.type,
             key.M, key.N, key.K);
    return it->second;
  }

  g_jkg_state.cache_misses++;
  log_info("Cache miss for kernel (type=%d, M=%zu, N=%zu, K=%zu)", key.type,
           key.M, key.N, key.K);
  return nullptr;
}

void insert_cached_kernel(const KernelCacheKey &key,
                          std::shared_ptr<jkg_kernel_impl_t> handle) {
  std::lock_guard<std::mutex> lock(g_jkg_state.cache_mutex);

  // Simple eviction: if cache is full, clear half
  if (g_jkg_state.kernel_cache.size() >= DEFAULT_CACHE_SIZE) {
    log_info("Cache full (%zu entries), clearing half",
             g_jkg_state.kernel_cache.size());
    auto it = g_jkg_state.kernel_cache.begin();
    size_t to_erase = g_jkg_state.kernel_cache.size() / 2;
    for (size_t i = 0; i < to_erase && it != g_jkg_state.kernel_cache.end();) {
      it = g_jkg_state.kernel_cache.erase(it);
    }
  }

  handle->is_cached = true;
  g_jkg_state.kernel_cache[key] = handle;
  log_info("Cached kernel (total=%zu)", g_jkg_state.kernel_cache.size());
}

/* ========================================================================== */
/* Kernel Name Generation */
/* ========================================================================== */

  std::string generate_kernel_name(jkg_kernel_type_t type,
                                   const jkg_kernel_params_t &params,
                                   jkg_isa_t isa) {
  char buf[MAX_KERNEL_NAME_LEN];
  const char *type_str = "unknown";

  switch (type) {
    case JKG_KERNEL_GEMM_TILE:
      type_str = "gemm_tile";
      break;
    case JKG_KERNEL_GEMM_BIAS:
      type_str = "gemm_bias";
      break;
    case JKG_KERNEL_GEMM_BIAS_RELU:
      type_str = "gemm_bias_relu";
      break;
    case JKG_KERNEL_GEMM_BIAS_ACT:
      type_str = "gemm_bias_act";
      break;
    case JKG_KERNEL_ELEMENTWISE_ADD:
      type_str = "elem_add";
      break;
    case JKG_KERNEL_ELEMENTWISE_MUL:
      type_str = "elem_mul";
      break;
    case JKG_KERNEL_ACTIVATION:
      type_str = "activation";
      break;
    default:
      break;
  }

  const char *isa_str = jkg_isa_name(isa);

  // For GEMM_BIAS_ACT, include the activation type in the name
  if (type == JKG_KERNEL_GEMM_BIAS_ACT) {
    const char *act_str = "none";
    switch (params.activation) {
      case JKG_ACT_RELU6: act_str = "relu6"; break;
      case JKG_ACT_GELU: act_str = "gelu"; break;
      case JKG_ACT_SWISH: act_str = "swish"; break;
      case JKG_ACT_TANH: act_str = "tanh"; break;
      case JKG_ACT_SIGMOID: act_str = "sigmoid"; break;
      default: act_str = "act"; break;
    }
    snprintf(buf, MAX_KERNEL_NAME_LEN, "%s_%s_%zux%zux%zu_%s",
             type_str, act_str, params.M, params.N, params.K, isa_str);
  } else {
    snprintf(buf, MAX_KERNEL_NAME_LEN, "%s_%zux%zux%zu_%s",
             type_str, params.M, params.N, params.K, isa_str);
  }

  return std::string(buf);
}

/* ========================================================================== */
/* Backend Selection */
/* ========================================================================== */

jkg_backend_t select_best_backend(jkg_isa_t isa) {
  // Priority: EVE > VectorClass > Highway > LLVM

#if JKG_HAS_EVE
  if (isa == JKG_ISA_AVX2 || isa == JKG_ISA_AVX512F) {
    log_info("Selected EVE backend for ISA");
    return JKG_BACKEND_EVE;
  }
#endif

#if JKG_HAS_VECTORCLASS && JKG_HAS_X86
  if (isa == JKG_ISA_AVX2 || isa == JKG_ISA_AVX512F) {
    log_info("Selected VectorClass backend for ISA");
    return JKG_BACKEND_VECTORCLASS;
  }
#endif

#if JKG_HAS_HIGHWAY
  log_info("Selected Highway backend for portable SIMD");
  return JKG_BACKEND_HIGHWAY;
#endif

  log_info("Selected pure LLVM backend");
  return JKG_BACKEND_LLVM;
}

std::unique_ptr<VectorizationBackend> create_backend(jkg_backend_t backend) {
  switch (backend) {
#if JKG_HAS_HIGHWAY
  case JKG_BACKEND_HIGHWAY:
    return std::make_unique<HighwayBackend>();
#endif
#if JKG_HAS_VECTORCLASS
  case JKG_BACKEND_VECTORCLASS:
    return std::make_unique<VectorClassBackend>();
#endif
#if JKG_HAS_EVE
  case JKG_BACKEND_EVE:
    return std::make_unique<EVEBackend>();
#endif
  default:
    return nullptr; // Pure LLVM
  }
}

/* ========================================================================== */
/* Target Triple and Features */
/* ========================================================================== */

std::string get_target_triple() {
#ifdef __x86_64__
  return "x86_64-unknown-linux-gnu";
#elif defined(__aarch64__)
  return "aarch64-unknown-linux-gnu";
#elif defined(__arm__)
  return "arm-unknown-linux-gnu";
#else
  return "unknown-unknown-unknown";
#endif
}

std::string get_target_cpu() {
#ifdef __x86_64__
  if (g_jkg_state.cpu_features.avx512)
    return "skylake-avx512";
  if (g_jkg_state.cpu_features.avx2)
    return "haswell";
  if (g_jkg_state.cpu_features.avx)
    return "sandybridge";
  return "x86-64";
#elif defined(__aarch64__)
  return "cortex-a72";
#else
  return "generic";
#endif
}

std::string get_target_features(jkg_isa_t isa) {
  std::string features;

  switch (isa) {
  case JKG_ISA_AVX512F:
    features = "+avx512f,+avx512bw,+avx512dq,+avx512vl";
    break;
  case JKG_ISA_AVX2:
    features = "+avx2,+fma";
    break;
  case JKG_ISA_AVX:
    features = "+avx";
    break;
  case JKG_ISA_SSE2:
    features = "+sse2";
    break;
  default:
    features = "";
  }

  return features;
}

} // namespace jkg_internal

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

using namespace jkg_internal;

int jkg_init(const jkg_config_t *config) {
  // Early check without logging to avoid issues when verbose is not set
  if (g_jkg_state.initialized) {
    return JKG_OK;
  }

  // Set default config first so we have verbose flag
  if (config) {
    g_jkg_state.config = *config;
  } else {
    g_jkg_state.config.target_isa = JKG_ISA_AUTO;
    g_jkg_state.config.backend = JKG_BACKEND_AUTO;
    g_jkg_state.config.enable_fma = 1;
    g_jkg_state.config.enable_prefetch = 1;
    g_jkg_state.config.enable_unroll = 1;
    g_jkg_state.config.unroll_factor = 4;
    g_jkg_state.config.cache_line_size = 64;
    g_jkg_state.config.optimization_level = DEFAULT_OPTIMIZATION_LEVEL;
    g_jkg_state.config.enable_kernel_cache = 1;
    g_jkg_state.config.verbose = 0;
  }

  log_info("Initializing JIT Kernel Generator");

  // Set default config if none provided
  if (config) {
    g_jkg_state.config = *config;
  } else {
    g_jkg_state.config.target_isa = JKG_ISA_AUTO;
    g_jkg_state.config.backend = JKG_BACKEND_AUTO;
    g_jkg_state.config.enable_fma = 1;
    g_jkg_state.config.enable_prefetch = 1;
    g_jkg_state.config.enable_unroll = 1;
    g_jkg_state.config.unroll_factor = 4;
    g_jkg_state.config.cache_line_size = 64;
    g_jkg_state.config.optimization_level = DEFAULT_OPTIMIZATION_LEVEL;
    g_jkg_state.config.enable_kernel_cache = 1;
    g_jkg_state.config.verbose = 0;
  }

  // Detect CPU features
  log_info("Detecting CPU features...");
  g_jkg_state.cpu_features = detect_cpu_features();
  g_jkg_state.cpu_info = detect_cpu_info();
  g_jkg_state.cache_info = read_cache_sysfs(0);
  g_jkg_state.available_isa_mask = detect_available_isa();

  log_info("CPU: %d cores, AVX=%d, AVX2=%d, AVX512=%d, AMX=%d",
           g_jkg_state.cpu_info.cores, g_jkg_state.cpu_features.avx,
           g_jkg_state.cpu_features.avx2, g_jkg_state.cpu_features.avx512,
           g_jkg_state.cpu_features.amx);

  // Initialize base components
  log_info("Initializing hardware introspection layer...");
  int jcore_ret = jcore_init();
  if (jcore_ret != JCORE_OK && jcore_ret != JCORE_ERR_CONFLICT) {
    log_error("jcore_init failed with code: %d", jcore_ret);
    return JKG_ERR_INTERNAL;
  }
  if (jcore_ret == JCORE_ERR_CONFLICT) {
    log_info("Hardware introspection already initialized");
  } else {
    log_info("Hardware introspection layer initialized");
  }

  log_info("Initializing ISA dispatch...");
  int dispatch_ret = jcore_init_dispatch();
  if (dispatch_ret != JCORE_OK && dispatch_ret != JCORE_ERR_CONFLICT) {
    log_error("jcore_init_dispatch failed with code: %d", dispatch_ret);
    jcore_shutdown();
    return JKG_ERR_INTERNAL;
  }
  if (dispatch_ret == JCORE_ERR_CONFLICT) {
    log_info("ISA dispatch already initialized");
  } else {
    log_info("ISA dispatch initialized");
  }

  // Initialize Kernel Fusion Engine
  bool kfe_was_initialized = kfe_is_initialized();
  if (!kfe_was_initialized) {
    log_info("Initializing Kernel Fusion Engine...");
    kfe_config_t kfe_cfg = {};
    kfe_cfg.num_threads = 0; // Auto
    kfe_cfg.enable_vectorization = 1;
    kfe_cfg.enable_cache_blocking = 1;
    kfe_cfg.enable_prefetch = 1;
    kfe_cfg.enable_kernel_autotuning = 1; // Enable - the benchmark is intentional
    kfe_cfg.verbose = 0;                  // Keep quiet during init
    kfe_cfg.workspace_size_mb = 256; // 256 MB

    int kfe_ret = kfe_init(&kfe_cfg);
    if (kfe_ret != KFE_OK) {
      // KFE is optional for JKG - we can work without it
      log_info("KFE initialization returned code %d - JKG will operate without "
               "KFE integration",
               kfe_ret);
      log_info("(This is not an error - KFE provides optimization but isn't "
               "required)");
    } else {
      log_info("Kernel Fusion Engine initialized successfully");
    }
  }

  // Adaptive Tuner is already initialized by KFE, so we skip it
  log_info("Adaptive tuner already initialized by KFE");

  // Note: LLVM initialization is deferred until first kernel generation
  // to avoid startup overhead
  log_info("LLVM initialization deferred until first kernel generation");

  g_jkg_state.initialized = true;
  log_info("JIT Kernel Generator initialized successfully");

  return JKG_OK;
}

void jkg_shutdown() {
  if (!g_jkg_state.initialized) {
    return;
  }

  log_info("Shutting down JIT Kernel Generator");

  // Clear kernel cache
  {
    std::lock_guard<std::mutex> lock(g_jkg_state.cache_mutex);
    g_jkg_state.kernel_cache.clear();
  }

  // Shutdown LLVM JIT
  if (g_jkg_state.jit) {
    g_jkg_state.jit.reset();
  }
  if (g_jkg_state.ts_context) {
    g_jkg_state.ts_context.reset();
  }

  // Reset LLVM initialization
  reset_llvm_init();

  // Shutdown dependencies in reverse order
  log_info("Shutting down base components");
  jcore_shutdown();

  g_jkg_state.initialized = false;
  log_info("JIT Kernel Generator shut down");
}

int jkg_is_initialized() { return g_jkg_state.initialized ? 1 : 0; }

int jkg_get_config(jkg_config_t *out_config) {
  if (!g_jkg_state.initialized) {
    return JKG_ERR_NOT_INITIALIZED;
  }
  if (!out_config) {
    return JKG_ERR_INVALID_ARG;
  }

  *out_config = g_jkg_state.config;
  return JKG_OK;
}

uint32_t jkg_get_available_isa() {
  // If not initialized, detect and return immediately
  if (!g_jkg_state.initialized) {
    return detect_available_isa();
  }
  return g_jkg_state.available_isa_mask;
}

const char *jkg_isa_name(jkg_isa_t isa) {
  switch (isa) {
  case JKG_ISA_SCALAR:
    return "scalar";
  case JKG_ISA_SSE2:
    return "sse2";
  case JKG_ISA_AVX:
    return "avx";
  case JKG_ISA_AVX2:
    return "avx2";
  case JKG_ISA_AVX512F:
    return "avx512f";
  case JKG_ISA_AVX512BW:
    return "avx512bw";
  case JKG_ISA_AMX:
    return "amx";
  case JKG_ISA_NEON:
    return "neon";
  case JKG_ISA_SVE:
    return "sve";
  case JKG_ISA_AUTO:
    return "auto";
  default:
    return "unknown";
  }
}

const char *jkg_backend_name(jkg_backend_t backend) {
  switch (backend) {
  case JKG_BACKEND_AUTO:
    return "auto";
  case JKG_BACKEND_HIGHWAY:
    return "highway";
  case JKG_BACKEND_VECTORCLASS:
    return "vectorclass";
  case JKG_BACKEND_EVE:
    return "eve";
  case JKG_BACKEND_LLVM:
    return "llvm";
  default:
    return "unknown";
  }
}

const char *jkg_strerror(int error) {
  switch (error) {
  case JKG_OK:
    return "Success";
  case JKG_ERR_NOT_INITIALIZED:
    return "JKG not initialized";
  case JKG_ERR_INVALID_ARG:
    return "Invalid argument";
  case JKG_ERR_NO_MEMORY:
    return "Out of memory";
  case JKG_ERR_INTERNAL:
    return "Internal error";
  case JKG_ERR_COMPILATION:
    return "Compilation failed";
  case JKG_ERR_UNSUPPORTED_ISA:
    return "Unsupported ISA";
  case JKG_ERR_CACHE_MISS:
    return "Cache miss";
  case JKG_ERR_LLVM_INIT:
    return "LLVM initialization failed";
  case JKG_ERR_NOT_FOUND:
    return "Kernel not found";
  case JKG_ERR_IR_INVALID:
    return "Invalid LLVM IR";
  default:
    return "Unknown error";
  }
}

int jkg_clear_cache() {
  if (!g_jkg_state.initialized) {
    return JKG_ERR_NOT_INITIALIZED;
  }

  std::lock_guard<std::mutex> lock(g_jkg_state.cache_mutex);
  size_t count = g_jkg_state.kernel_cache.size();
  g_jkg_state.kernel_cache.clear();

  log_info("Cleared %zu cached kernels", count);
  return JKG_OK;
}

int jkg_get_cache_stats(size_t *out_cached, size_t *out_hits,
                        size_t *out_misses) {
  if (!g_jkg_state.initialized) {
    // Return zeros for uninitialized state instead of error
    if (out_cached)
      *out_cached = 0;
    if (out_hits)
      *out_hits = 0;
    if (out_misses)
      *out_misses = 0;
    return JKG_OK;
  }

  if (out_cached) {
    std::lock_guard<std::mutex> lock(g_jkg_state.cache_mutex);
    *out_cached = g_jkg_state.kernel_cache.size();
  }
  if (out_hits)
    *out_hits = g_jkg_state.cache_hits.load();
  if (out_misses)
    *out_misses = g_jkg_state.cache_misses.load();

  return JKG_OK;
}

int jkg_get_optimal_tile_sizes(jkg_isa_t target_isa, size_t *out_M,
                               size_t *out_N, size_t *out_K) {
  if (!g_jkg_state.initialized) {
    return JKG_ERR_NOT_INITIALIZED;
  }
  if (!out_M || !out_N || !out_K) {
    return JKG_ERR_INVALID_ARG;
  }

  jkg_isa_t isa = (target_isa == JKG_ISA_AUTO)
                      ? select_best_isa(g_jkg_state.available_isa_mask)
                      : target_isa;

  compute_optimal_tile_sizes(isa, g_jkg_state.cache_info, *out_M, *out_N,
                             *out_K);

  return JKG_OK;
}

const char *jkg_get_system_info() {
  static char buf[512];

  if (!g_jkg_state.initialized) {
    return "JKG not initialized";
  }

  snprintf(buf, sizeof(buf),
           "CPU: %d cores, L1d=%dKB, L2=%dKB, L3=%dKB\n"
           "ISA: AVX=%d, AVX2=%d, AVX-512=%d, AMX=%d\n"
           "Cache: %zu kernels, %zu hits, %zu misses\n"
           "Generated: %zu kernels",
           g_jkg_state.cpu_info.cores, g_jkg_state.cpu_info.l1d_kb,
           g_jkg_state.cpu_info.l2_kb, g_jkg_state.cpu_info.l3_kb,
           g_jkg_state.cpu_features.avx, g_jkg_state.cpu_features.avx2,
           g_jkg_state.cpu_features.avx512, g_jkg_state.cpu_features.amx,
           g_jkg_state.kernel_cache.size(), g_jkg_state.cache_hits.load(),
           g_jkg_state.cache_misses.load(),
           g_jkg_state.kernels_generated.load());

  return buf;
}