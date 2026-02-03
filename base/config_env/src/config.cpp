// src/config.cpp
#include "config.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>
#include <cerrno>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace jcore
{
  namespace config
  {

    // ------------------------- Helpers -------------------------
    static inline std::string ToLower(std::string s) noexcept
    {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                     { return static_cast<char>(std::tolower(c)); });
      return s;
    }

    static inline std::size_t HardwareConcurrencyOrDefault() noexcept
    {
      unsigned hc = std::thread::hardware_concurrency();
      return (hc == 0u ? 1u : static_cast<std::size_t>(hc));
    }

    // parse helpers
    static bool ParseSizeT(const std::string &s, std::size_t &out) noexcept
    {
      if (s.empty())
        return false;
      char *endptr = nullptr;
      errno = 0;
      unsigned long val = strtoul(s.c_str(), &endptr, 10);
      if (errno != 0 || endptr == s.c_str() || *endptr != '\0')
        return false;
      out = static_cast<std::size_t>(val);
      return true;
    }

    static bool ParseInt(const std::string &s, int &out) noexcept
    {
      if (s.empty())
        return false;
      char *endptr = nullptr;
      errno = 0;
      long val = strtol(s.c_str(), &endptr, 10);
      if (errno != 0 || endptr == s.c_str() || *endptr != '\0')
        return false;
      out = static_cast<int>(val);
      return true;
    }

    // ------------------------- ParseEnv -------------------------
    Result Config::ParseEnv() noexcept
    {
      // Read environment variables prefixed by JCORE_
      // Threads
      const char *e_threads = std::getenv("JCORE_THREADS");
      if (e_threads)
      {
        std::string s = e_threads;
        std::size_t v;
        if (ParseSizeT(s, v))
        {
          threads = v;
        }
        else
        {
          return Result(false, "Invalid JCORE_THREADS value: " + s);
        }
      }

      // Backend
      const char *e_backend = std::getenv("JCORE_BACKEND");
      if (e_backend)
      {
        std::string s = ToLower(e_backend);
        if (s == "tbb")
          backend = Backend::TBB;
        else if (s == "omp")
          backend = Backend::OpenMP;
        else if (s == "serial")
          backend = Backend::Serial;
        else if (s == "auto")
          backend = Backend::Auto;
        else
          return Result(false, "Invalid JCORE_BACKEND: " + s);
      }

      // Memory mode
      const char *e_mem = std::getenv("JCORE_MEMORY_MODE");
      if (e_mem)
      {
        std::string s = ToLower(e_mem);
        if (s == "default")
          memory_mode = MemoryMode::Default;
        else if (s == "hugepages")
          memory_mode = MemoryMode::HugePages;
        else if (s == "mlock")
          memory_mode = MemoryMode::MLock;
        else
          return Result(false, "Invalid JCORE_MEMORY_MODE: " + s);
      }

      // Affinity
      const char *e_aff = std::getenv("JCORE_AFFINITY");
      if (e_aff)
      {
        std::string s = ToLower(e_aff);
        if (s == "none")
          affinity = AffinityMode::None;
        else if (s == "auto")
          affinity = AffinityMode::Auto;
        else
          return Result(false, "Invalid JCORE_AFFINITY: " + s);
      }

      // NUMA node
      const char *e_numa = std::getenv("JCORE_NUMA_NODE");
      if (e_numa)
      {
        int v;
        if (!ParseInt(std::string(e_numa), v))
          return Result(false, "Invalid JCORE_NUMA_NODE");
        numa_node = v;
      }

      // Log level
      const char *e_log = std::getenv("JCORE_LOG_LEVEL");
      if (e_log)
      {
        std::string s = ToLower(e_log);
        if (s == "debug")
          log_level = LogLevel::Debug;
        else if (s == "info")
          log_level = LogLevel::Info;
        else if (s == "warn")
          log_level = LogLevel::Warn;
        else if (s == "error")
          log_level = LogLevel::Error;
        else
          return Result(false, "Invalid JCORE_LOG_LEVEL: " + s);
      }

      // Debug
      const char *e_dbg = std::getenv("JCORE_DEBUG");
      if (e_dbg)
      {
        std::string s = ToLower(e_dbg);
        if (s == "1" || s == "true" || s == "yes")
          debug = true;
        else if (s == "0" || s == "false" || s == "no")
          debug = false;
        else
          return Result(false, "Invalid JCORE_DEBUG: " + s);
      }

      // Record raw source for diagnostics
      std::ostringstream os;
      os << "env:";
      if (e_threads)
        os << " threads=" << e_threads;
      if (e_backend)
        os << " backend=" << e_backend;
      if (e_mem)
        os << " mem=" << e_mem;
      if (e_aff)
        os << " affinity=" << e_aff;
      if (e_numa)
        os << " numa=" << e_numa;
      if (e_log)
        os << " log=" << e_log;
      if (e_dbg)
        os << " debug=" << e_dbg;
      raw_from_env = os.str();

      return Result(true, "");
    }

    // ------------------------- ParseArgs -------------------------
    Result Config::ParseArgs(int argc, char **argv) noexcept
    {
      // Simple manual parse. Flags override env vars.
      // Supported forms:
      //   --threads N
      //   --backend tbb|omp|serial|auto
      //   --mem default|hugepages|mlock
      //   --affinity auto|none
      //   --numa N
      //   --log debug|info|warn|error
      //   --debug   (sets debug=true)
      //   --help    (returns ok=false with help message)
      raw_from_args.clear();
      std::ostringstream os;
      for (int i = 1; i < argc; ++i)
      {
        std::string a(argv[i]);
        if (a == "--help" || a == "-h")
        {
          os << "Usage: [--threads N] [--backend tbb|omp|serial|auto] [--mem default|hugepages|mlock] "
             << "[--affinity auto|none] [--numa N] [--log debug|info|warn|error] [--debug]\n";
          raw_from_args = os.str();
          return Result(false, raw_from_args);
        }
        else if (a == "--threads" && i + 1 < argc)
        {
          std::size_t v;
          if (!ParseSizeT(argv[++i], v))
            return Result(false, "Invalid --threads value");
          threads = v;
          os << " threads=" << v;
        }
        else if (a == "--backend" && i + 1 < argc)
        {
          std::string v = ToLower(argv[++i]);
          if (v == "tbb")
            backend = Backend::TBB;
          else if (v == "omp")
            backend = Backend::OpenMP;
          else if (v == "serial")
            backend = Backend::Serial;
          else if (v == "auto")
            backend = Backend::Auto;
          else
            return Result(false, "Invalid --backend: " + v);
          os << " backend=" << v;
        }
        else if (a == "--mem" && i + 1 < argc)
        {
          std::string v = ToLower(argv[++i]);
          if (v == "default")
            memory_mode = MemoryMode::Default;
          else if (v == "hugepages")
            memory_mode = MemoryMode::HugePages;
          else if (v == "mlock")
            memory_mode = MemoryMode::MLock;
          else
            return Result(false, "Invalid --mem: " + v);
          os << " mem=" << v;
        }
        else if (a == "--affinity" && i + 1 < argc)
        {
          std::string v = ToLower(argv[++i]);
          if (v == "auto")
            affinity = AffinityMode::Auto;
          else if (v == "none")
            affinity = AffinityMode::None;
          else
            return Result(false, "Invalid --affinity: " + v);
          os << " affinity=" << v;
        }
        else if (a == "--numa" && i + 1 < argc)
        {
          int v;
          if (!ParseInt(argv[++i], v))
            return Result(false, "Invalid --numa value");
          numa_node = v;
          os << " numa=" << v;
        }
        else if (a == "--log" && i + 1 < argc)
        {
          std::string v = ToLower(argv[++i]);
          if (v == "debug")
            log_level = LogLevel::Debug;
          else if (v == "info")
            log_level = LogLevel::Info;
          else if (v == "warn")
            log_level = LogLevel::Warn;
          else if (v == "error")
            log_level = LogLevel::Error;
          else
            return Result(false, "Invalid --log: " + v);
          os << " log=" << v;
        }
        else if (a == "--debug")
        {
          debug = true;
          os << " debug=1";
        }
        else
        {
          // ignore unknown flags but record them
          os << " unknown(" << a << ")";
        }
      }
      raw_from_args = os.str();
      return Result(true, "");
    }

    // ------------------------- Validate -------------------------
    Result Config::Validate() const noexcept
    {
      // threads must be reasonable
      if (threads == 0)
      {
        // ok (means auto)
      }
      else
      {
        const std::size_t max_threads = HardwareConcurrencyOrDefault() * 256; // arbitrary upper bound
        if (threads > max_threads)
        {
          return Result(false, "threads value too large");
        }
      }

      if (numa_node < -1)
      {
        return Result(false, "numa_node invalid");
      }

      // memory_mode ok
      // backend ok

      return Result(true, "");
    }

    // ------------------------- ApplyRuntimeHints -------------------------
    Result Config::ApplyRuntimeHints() const noexcept
    {
      // Apply thread hint (best-effort): set environment variable to inform other libraries
      // We do not actually change thread pool sizes here.
      if (threads > 0)
      {
        std::string s = std::to_string(threads);
        // set common env vars used by libraries (best-effort)
        // Note: setenv may fail but we ignore non-fatal errors.
        if (setenv("JCORE_THREADS", s.c_str(), 1) != 0)
        {
          // fall-through, not fatal
        }
        // also try OMP_NUM_THREADS and TBB_THREADING_CONTROL? TBB uses global_control programmatically, so env is only hint.
        setenv("OMP_NUM_THREADS", s.c_str(), 1);
        setenv("TBB_NUM_THREADS", s.c_str(), 1);
      }

      // Memory mode: best-effort. For hugepages, user must ensure system configured; we can try to mmap with MAP_HUGETLB on Linux (requires privileges).
#if defined(__linux__)
      if (memory_mode == MemoryMode::HugePages)
      {
        // We won't actually reserve large buffers here (that could be heavy). Instead, we try to indicate success only.
        // Real hugepage allocation should be done by the consumer using mmap with MAP_HUGETLB and proper size.
        // Return info as a warning if not root or if MAP_HUGETLB fails when tried later.
      }
      else if (memory_mode == MemoryMode::MLock)
      {
        // Optionally attempt to mlockall; this requires privileges and can fail.
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        {
          // Non-fatal; return info when debug
          if (debug)
          {
            std::ostringstream os;
            os << "mlockall failed: " << strerror(errno);
            return Result(false, os.str());
          }
        }
      }
#endif

      // Affinity: if requested, apply a simple round-robin CPU pinning for current thread (best-effort).
#if defined(__linux__)
      if (affinity == AffinityMode::Auto)
      {
        // Try to pin calling thread to a CPU derived from process id to spread processes.
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        const unsigned int hc = HardwareConcurrencyOrDefault();
        if (hc == 0)
        {
          return Result(false, "Hardware concurrency unknown; cannot apply affinity");
        }
        // Simple heuristic: pick cpu = (pid % hc)
        pid_t pid = getpid();
        unsigned int cpu = static_cast<unsigned int>(pid % static_cast<pid_t>(hc));
        CPU_SET(cpu, &cpus);
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpus) != 0)
        {
          std::ostringstream os;
          os << "sched_setaffinity failed: " << strerror(errno);
          return Result(false, os.str());
        }
      }
#else
      (void)affinity; // silence unused
#endif

      // NUMA node: best-effort (not implemented here). Consumers should call numa APIs as needed.

      return Result(true, "");
    }

    // ------------------------- ToString -------------------------
    std::string Config::ToString() const noexcept
    {
      std::ostringstream os;
      os << "Config{ threads=" << threads;
      os << " backend=";
      switch (backend)
      {
      case Backend::Auto:
        os << "auto";
        break;
      case Backend::TBB:
        os << "tbb";
        break;
      case Backend::OpenMP:
        os << "omp";
        break;
      case Backend::Serial:
        os << "serial";
        break;
      }
      os << " memory_mode=";
      switch (memory_mode)
      {
      case MemoryMode::Default:
        os << "default";
        break;
      case MemoryMode::HugePages:
        os << "hugepages";
        break;
      case MemoryMode::MLock:
        os << "mlock";
        break;
      }
      os << " affinity=" << (affinity == AffinityMode::Auto ? "auto" : "none");
      os << " numa_node=" << numa_node;
      os << " log_level=";
      switch (log_level)
      {
      case LogLevel::Debug:
        os << "debug";
        break;
      case LogLevel::Info:
        os << "info";
        break;
      case LogLevel::Warn:
        os << "warn";
        break;
      case LogLevel::Error:
        os << "error";
        break;
      }
      os << " debug=" << (debug ? "1" : "0") << " }";
      if (!raw_from_env.empty())
        os << " [" << raw_from_env << "]";
      if (!raw_from_args.empty())
        os << " [" << raw_from_args << "]";
      return os.str();
    }

    // ------------------------- C API -------------------------
    extern "C"
    {

      static inline Backend IntToBackend(int v) noexcept
      {
        switch (v)
        {
        case 1:
          return Backend::TBB;
        case 2:
          return Backend::OpenMP;
        case 3:
          return Backend::Serial;
        default:
          return Backend::Auto;
        }
      }
      static inline int BackendToInt(Backend b) noexcept
      {
        switch (b)
        {
        case Backend::TBB:
          return 1;
        case Backend::OpenMP:
          return 2;
        case Backend::Serial:
          return 3;
        case Backend::Auto:
        default:
          return 0;
        }
      }
      static inline MemoryMode IntToMem(int v) noexcept
      {
        switch (v)
        {
        case 1:
          return MemoryMode::HugePages;
        case 2:
          return MemoryMode::MLock;
        default:
          return MemoryMode::Default;
        }
      }
      static inline int MemToInt(MemoryMode m) noexcept
      {
        switch (m)
        {
        case MemoryMode::HugePages:
          return 1;
        case MemoryMode::MLock:
          return 2;
        default:
          return 0;
        }
      }
      static inline AffinityMode IntToAff(int v) noexcept
      {
        return (v == 1 ? AffinityMode::Auto : AffinityMode::None);
      }
      static inline int AffToInt(AffinityMode a) noexcept
      {
        return (a == AffinityMode::Auto ? 1 : 0);
      }
      static inline int LogToInt(LogLevel l) noexcept
      {
        switch (l)
        {
        case LogLevel::Debug:
          return 0;
        case LogLevel::Info:
          return 1;
        case LogLevel::Warn:
          return 2;
        case LogLevel::Error:
          return 3;
        default:
          return 1;
        }
      }

      int ffm_config_init_from_env_and_args(FFMConfig *out_cfg, int argc, char **argv)
      {
        if (!out_cfg)
          return -1;
        Config cfg;
        Result r = cfg.ParseEnv();
        if (!r.ok)
          return -2;
        r = cfg.ParseArgs(argc, argv);
        if (!r.ok)
        {
          // If ParseArgs returned help message (raw_from_args non-empty), still copy current cfg but return code 1.
          if (!cfg.raw_from_args.empty())
          {
            // fill out partial config
          }
          else
          {
            return -3;
          }
        }
        r = cfg.Validate();
        if (!r.ok)
          return -4;
        // fill out out_cfg
        out_cfg->threads = cfg.threads;
        out_cfg->backend = BackendToInt(cfg.backend);
        out_cfg->memory_mode = MemToInt(cfg.memory_mode);
        out_cfg->affinity = AffToInt(cfg.affinity);
        out_cfg->numa_node = cfg.numa_node;
        out_cfg->log_level = LogToInt(cfg.log_level);
        out_cfg->debug = cfg.debug ? 1 : 0;
        return 0;
      }

      void ffm_config_free(FFMConfig *cfg)
      {
        // no dynamic resources owned; present for API symmetry
        (void)cfg;
      }

      int ffm_config_apply_runtime(const FFMConfig *cfg)
      {
        if (!cfg)
          return -1;
        Config cppcfg;
        cppcfg.threads = cfg->threads;
        cppcfg.backend = IntToBackend(cfg->backend);
        cppcfg.memory_mode = IntToMem(cfg->memory_mode);
        cppcfg.affinity = IntToAff(cfg->affinity);
        cppcfg.numa_node = cfg->numa_node;
        cppcfg.log_level = LogLevel::Info;
        cppcfg.debug = (cfg->debug != 0);
        Result r = cppcfg.ApplyRuntimeHints();
        return r.ok ? 0 : -2;
      }

    } // extern "C"

  } // namespace config
} // namespace jcore
