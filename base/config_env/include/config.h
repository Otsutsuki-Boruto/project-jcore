// include/config.h
#ifndef JC_CORE_CONFIG_H_
#define JC_CORE_CONFIG_H_

// Configuration & Env Controller - Project JCore
//
// Responsibilities:
//  - Read configuration from environment variables and from command-line flags.
//  - Validate values, provide defaults, expose a simple API used by other components.
//  - Best-effort runtime actions: set thread count hints, apply CPU affinity (Linux).
//  - Provide C-compatible FFM API for minimal integration.
//
// Usage:
//   jcore::config::Config cfg;
//   cfg.ParseEnv();              // read environment
//   cfg.ParseArgs(argc, argv);   // override with flags
//   cfg.ApplyRuntimeHints();     // best-effort actions
//   std::cout << cfg.ToString() << std::endl;
//
// Env vars (prefix JCORE_):
//   JCORE_THREADS        - number (0 means auto)
//   JCORE_BACKEND        - "tbb"/"omp"/"serial"/"auto"
//   JCORE_MEMORY_MODE    - "default"/"hugepages"/"mlock"
//   JCORE_AFFINITY       - "auto"/"none" (auto = try to pin to socket/core round-robin)
//   JCORE_NUMA_NODE      - integer >=0 or -1 for none
//   JCORE_LOG_LEVEL      - "debug"/"info"/"warn"/"error"
//   JCORE_DEBUG          - "1"/"0"
//
// Command-line flags (override env):
//   --threads N
//   --backend tbb|omp|serial|auto
//   --mem default|hugepages|mlock
//   --affinity auto|none
//   --numa N
//   --log lvl
//   --debug  (flag sets debug=true)
//
// C API (FFM):
//   struct FFMConfig C-compatible struct, and functions ffm_config_init, ffm_config_free.
//
// Thread-safety: Config object is not internally synchronized. Read/modify during initialization only.

#include <cstddef>
#include <string>

namespace jcore
{
  namespace config
  {

    // Backend selection
    enum class Backend
    {
      Auto,
      TBB,
      OpenMP,
      Serial
    };

    // Memory mode
    enum class MemoryMode
    {
      Default,
      HugePages,
      MLock
    };

    // Affinity hint
    enum class AffinityMode
    {
      None,
      Auto
    };

    // Simple log level
    enum class LogLevel
    {
      Debug,
      Info,
      Warn,
      Error
    };

    // Result/Error container
    struct Result
    {
      bool ok;
      std::string message;
      Result(bool b = true, std::string m = "") : ok(b), message(std::move(m)) {}
    };

    // Primary configuration container
    struct Config
    {
      // Tunable options
      std::size_t threads = 0; // 0 => auto (hardware_concurrency)
      Backend backend = Backend::Auto;
      MemoryMode memory_mode = MemoryMode::Default;
      AffinityMode affinity = AffinityMode::None;
      int numa_node = -1; // -1 => unspecified
      LogLevel log_level = LogLevel::Info;
      bool debug = false;

      // Raw sources (read-only)
      std::string raw_from_env;  // last parsed env string debug
      std::string raw_from_args; // last parsed args string debug

      // Parse environment variables. Returns Result with error message if parsing failed.
      Result ParseEnv() noexcept;

      // Parse command-line args (argc/argv). Flags override env vars. Returns Result.
      Result ParseArgs(int argc, char **argv) noexcept;

      // Validate fields and normalize values. Does not perform runtime actions.
      Result Validate() const noexcept;

      // Apply best-effort runtime hints (e.g., set CPU affinity or lock memory).
      // Returns Result; non-fatal failures are returned as ok=false with message.
      Result ApplyRuntimeHints() const noexcept;

      // Pretty string representation for logging.
      std::string ToString() const noexcept;
    };

    // ---------- C-compatible FFM API ----------
    // Minimal C struct corresponding to Config
    extern "C"
    {
      struct FFMConfig
      {
        size_t threads;
        int backend;     // 0=auto,1=tbb,2=omp,3=serial
        int memory_mode; // 0=default,1=hugepages,2=mlock
        int affinity;    // 0=none,1=auto
        int numa_node;
        int log_level; // 0=debug,1=info,2=warn,3=error
        int debug;     // 0/1
      };

      // Initialize FFMConfig from env and args. Returns 0 on success, non-zero otherwise.
      int ffm_config_init_from_env_and_args(FFMConfig *out_cfg, int argc, char **argv);

      // Free any resources (no-op for now, provided for API symmetry)
      void ffm_config_free(FFMConfig *cfg);

      // Apply runtime hints for the given FFMConfig. Returns 0 on success.
      int ffm_config_apply_runtime(const FFMConfig *cfg);
    } // extern "C"

  } // namespace config
} // namespace jcore

#endif // JC_CORE_CONFIG_H_
