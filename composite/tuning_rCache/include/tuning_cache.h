// composite/tuning_cache/include/tuning_cache.h
#ifndef JCORE_TUNING_CACHE_H_
#define JCORE_TUNING_CACHE_H_

// Tuning Result Cache System - Project JCore
//
// Purpose:
//   Persists auto-tuning outcomes (JSON/binary) to avoid re-benchmarking.
//   Stores best kernel selection per hardware signature + operation shape.
//
// Features:
//   - Persistent storage across runs
//   - Thread-safe concurrent access
//   - JSON (human-readable) and binary (compact) formats
//   - Hardware signature validation
//   - Cache versioning and invalidation
//   - Statistics tracking (hit rate, misses)
//   - Export/import for deployment
//
// Dependencies:
//   - Configuration & Env Controller (config.h)
//   - Adaptive Kernel AutoTuner (adaptive_tuner.h)
//   - Standard C++17 features
//
// Design:
//   - Google C++ style guidelines
//   - RAII resource management
//   - Error codes for recoverable errors
//   - Thread-safe with internal mutex
//   - FFM-compatible C API

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>

namespace jcore
{
  namespace tuning_cache
  {

    // Cache version - increment when tuning algorithm changes to invalidate old caches
    constexpr uint32_t CACHE_VERSION = 1;

    // Maximum kernel name length
    constexpr size_t MAX_KERNEL_NAME_LEN = 256;

    // Status codes
    enum class Status
    {
      OK = 0,
      ERR_NOT_INITIALIZED,
      ERR_INVALID_ARG,
      ERR_NO_MEMORY,
      ERR_IO_FAILURE,
      ERR_PARSE_FAILURE,
      ERR_VERSION_MISMATCH,
      ERR_HARDWARE_MISMATCH,
      ERR_NOT_FOUND,
      ERR_ALREADY_EXISTS,
      ERR_LOCK_FAILURE,
      ERR_INTERNAL
    };

    // Storage format
    enum class Format
    {
      JSON,  // Human-readable, larger files
      BINARY // Compact, faster I/O
    };

    // Hardware signature - identifies CPU capabilities
    struct HardwareSignature
    {
      uint64_t features;    // Bitfield: AVX2=1, AVX512=2, AMX=4, etc.
      uint32_t cores;       // Physical core count
      uint32_t l1_cache_kb; // L1 data cache size
      uint32_t l2_cache_kb; // L2 cache size
      uint32_t l3_cache_kb; // L3 cache size
      int32_t numa_nodes;   // NUMA node count (-1 if unknown)

      HardwareSignature();
      bool operator==(const HardwareSignature &other) const noexcept;
      bool operator!=(const HardwareSignature &other) const noexcept;

      // Generate hash for this signature
      uint64_t Hash() const noexcept;

      // Serialize to string
      std::string ToString() const noexcept;
    };

    // Operation signature - identifies computation shape
    struct OperationSignature
    {
      size_t M, N, K;   // Matrix dimensions
      size_t threads;   // Thread count used
      size_t tile_size; // Tile size hint
      uint32_t op_type; // 0=GEMM, future: 1=Conv, 2=Pooling, etc.

      OperationSignature();
      OperationSignature(size_t m, size_t n, size_t k,
                         size_t thr = 1, size_t tile = 0, uint32_t type = 0);

      bool operator==(const OperationSignature &other) const noexcept;
      bool operator!=(const OperationSignature &other) const noexcept;

      // Generate hash for this signature
      uint64_t Hash() const noexcept;

      // Serialize to string
      std::string ToString() const noexcept;
    };

    // Tuning result entry
    struct TuningEntry
    {
      OperationSignature op_sig;
      char kernel_name[MAX_KERNEL_NAME_LEN]; // Best kernel name
      double performance_gflops;             // Measured performance
      double benchmark_time_usec;            // Time taken for benchmark
      uint64_t timestamp;                    // Unix epoch when cached
      uint32_t cache_version;                // Cache format version

      TuningEntry();
      TuningEntry(const OperationSignature &sig, const char *name,
                  double gflops, double time_usec);
    };

    // Cache statistics
    struct CacheStats
    {
      size_t total_entries;
      size_t hits;
      size_t misses;
      size_t evictions;
      double hit_rate;

      CacheStats();
      std::string ToString() const noexcept;
    };

    // Cache configuration
    struct CacheConfig
    {
      std::string cache_dir;  // Directory for cache files
      Format format;          // Storage format
      size_t max_entries;     // Max entries (0 = unlimited)
      bool validate_hardware; // Validate HW signature on load
      bool auto_save;         // Auto-save on modifications
      bool thread_safe;       // Enable internal locking

      CacheConfig();

      // Load config from jcore::config::Config
      void LoadFromEnv() noexcept;
    };

    // Main cache interface
    class TuningCache
    {
    public:
      TuningCache();
      ~TuningCache();

      // Non-copyable, movable
      TuningCache(const TuningCache &) = delete;
      TuningCache &operator=(const TuningCache &) = delete;
      TuningCache(TuningCache &&) noexcept;
      TuningCache &operator=(TuningCache &&) noexcept;

      // Initialize cache with configuration
      Status Init(const CacheConfig &config) noexcept;

      // Shutdown and cleanup
      void Shutdown() noexcept;

      // Query: lookup cached tuning result
      Status Query(const OperationSignature &op_sig, TuningEntry &out_entry) noexcept;

      // Insert: store tuning result
      Status Insert(const TuningEntry &entry) noexcept;

      // Remove: delete specific entry
      Status Remove(const OperationSignature &op_sig) noexcept;

      // Clear: remove all entries
      Status Clear() noexcept;

      // Save: persist cache to disk
      Status Save() noexcept;

      // Load: read cache from disk
      Status Load() noexcept;

      // Export: write cache to specific file
      Status Export(const std::string &filepath, Format format) noexcept;

      // Import: read cache from specific file
      Status Import(const std::string &filepath) noexcept;

      // Get statistics
      CacheStats GetStats() const noexcept;

      // Get hardware signature
      HardwareSignature GetHardwareSignature() const noexcept;

      // Check if initialized
      bool IsInitialized() const noexcept;

    private:
      struct Impl;
      std::unique_ptr<Impl> impl_;
    };

    // Utility: convert status to string
    const char *StatusToString(Status s) noexcept;

    // Utility: detect current hardware signature
    HardwareSignature DetectHardware() noexcept;

    // ---------- C-compatible FFM API ----------
    extern "C"
    {
      // Opaque handle
      typedef struct tc_cache_t tc_cache_t;

      // Status codes (matches Status enum)
      typedef enum
      {
        TC_OK = 0,
        TC_ERR_NOT_INITIALIZED = 1,
        TC_ERR_INVALID_ARG = 2,
        TC_ERR_NO_MEMORY = 3,
        TC_ERR_IO_FAILURE = 4,
        TC_ERR_PARSE_FAILURE = 5,
        TC_ERR_VERSION_MISMATCH = 6,
        TC_ERR_HARDWARE_MISMATCH = 7,
        TC_ERR_NOT_FOUND = 8,
        TC_ERR_ALREADY_EXISTS = 9,
        TC_ERR_LOCK_FAILURE = 10,
        TC_ERR_INTERNAL = 11
      } tc_status_t;

      // Format
      typedef enum
      {
        TC_FORMAT_JSON = 0,
        TC_FORMAT_BINARY = 1
      } tc_format_t;

      // C structs
      typedef struct
      {
        size_t M, N, K;
        size_t threads;
        size_t tile_size;
        uint32_t op_type;
      } tc_op_signature_t;

      typedef struct
      {
        tc_op_signature_t op_sig;
        char kernel_name[256];
        double performance_gflops;
        double benchmark_time_usec;
        uint64_t timestamp;
        uint32_t cache_version;
      } tc_tuning_entry_t;

      typedef struct
      {
        char cache_dir[512];
        tc_format_t format;
        size_t max_entries;
        int validate_hardware;
        int auto_save;
        int thread_safe;
      } tc_cache_config_t;

      typedef struct
      {
        size_t total_entries;
        size_t hits;
        size_t misses;
        size_t evictions;
        double hit_rate;
      } tc_cache_stats_t;

      // Create cache instance
      tc_cache_t *tc_cache_create(void);

      // Initialize cache
      tc_status_t tc_cache_init(tc_cache_t *cache, const tc_cache_config_t *config);

      // Shutdown cache
      void tc_cache_shutdown(tc_cache_t *cache);

      // Destroy cache instance
      void tc_cache_destroy(tc_cache_t *cache);

      // Query entry
      tc_status_t tc_cache_query(tc_cache_t *cache, const tc_op_signature_t *op_sig,
                                 tc_tuning_entry_t *out_entry);

      // Insert entry
      tc_status_t tc_cache_insert(tc_cache_t *cache, const tc_tuning_entry_t *entry);

      // Remove entry
      tc_status_t tc_cache_remove(tc_cache_t *cache, const tc_op_signature_t *op_sig);

      // Clear all
      tc_status_t tc_cache_clear(tc_cache_t *cache);

      // Save to disk
      tc_status_t tc_cache_save(tc_cache_t *cache);

      // Load from disk
      tc_status_t tc_cache_load(tc_cache_t *cache);

      // Get statistics
      tc_status_t tc_cache_get_stats(tc_cache_t *cache, tc_cache_stats_t *out_stats);

      // Status to string
      const char *tc_status_str(tc_status_t s);

      // Initialize default config
      void tc_cache_config_init_default(tc_cache_config_t *config);

    } // extern "C"

  } // namespace tuning_cache
} // namespace jcore

#endif // JCORE_TUNING_CACHE_H_