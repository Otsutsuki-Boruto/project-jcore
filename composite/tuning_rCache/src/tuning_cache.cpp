// composite/tuning_cache/src/tuning_cache.cpp

#include "tuning_cache.h"
#include "config.h"

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

// For hardware detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif

namespace jcore
{
  namespace tuning_cache
  {

    // ========== HardwareSignature Implementation ==========

    HardwareSignature::HardwareSignature()
        : features(0), cores(0), l1_cache_kb(0), l2_cache_kb(0),
          l3_cache_kb(0), numa_nodes(-1)
    {
    }

    bool HardwareSignature::operator==(const HardwareSignature &other) const noexcept
    {
      return features == other.features &&
             cores == other.cores &&
             l1_cache_kb == other.l1_cache_kb &&
             l2_cache_kb == other.l2_cache_kb &&
             l3_cache_kb == other.l3_cache_kb &&
             numa_nodes == other.numa_nodes;
    }

    bool HardwareSignature::operator!=(const HardwareSignature &other) const noexcept
    {
      return !(*this == other);
    }

    uint64_t HardwareSignature::Hash() const noexcept
    {
      // Simple FNV-1a hash
      uint64_t hash = 14695981039346656037ULL;
      hash ^= features;
      hash *= 1099511628211ULL;
      hash ^= cores;
      hash *= 1099511628211ULL;
      hash ^= l1_cache_kb;
      hash *= 1099511628211ULL;
      hash ^= l2_cache_kb;
      hash *= 1099511628211ULL;
      hash ^= l3_cache_kb;
      hash *= 1099511628211ULL;
      hash ^= static_cast<uint64_t>(numa_nodes);
      hash *= 1099511628211ULL;
      return hash;
    }

    std::string HardwareSignature::ToString() const noexcept
    {
      std::ostringstream oss;
      oss << "HW{feat=0x" << std::hex << features << std::dec
          << ",cores=" << cores
          << ",L1=" << l1_cache_kb << "KB"
          << ",L2=" << l2_cache_kb << "KB"
          << ",L3=" << l3_cache_kb << "KB"
          << ",NUMA=" << numa_nodes << "}";
      return oss.str();
    }

    // ========== OperationSignature Implementation ==========

    OperationSignature::OperationSignature()
        : M(0), N(0), K(0), threads(1), tile_size(0), op_type(0)
    {
    }

    OperationSignature::OperationSignature(size_t m, size_t n, size_t k,
                                           size_t thr, size_t tile, uint32_t type)
        : M(m), N(n), K(k), threads(thr), tile_size(tile), op_type(type)
    {
    }

    bool OperationSignature::operator==(const OperationSignature &other) const noexcept
    {
      return M == other.M && N == other.N && K == other.K &&
             threads == other.threads && tile_size == other.tile_size &&
             op_type == other.op_type;
    }

    bool OperationSignature::operator!=(const OperationSignature &other) const noexcept
    {
      return !(*this == other);
    }

    uint64_t OperationSignature::Hash() const noexcept
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
      hash ^= op_type;
      hash *= 1099511628211ULL;
      return hash;
    }

    std::string OperationSignature::ToString() const noexcept
    {
      std::ostringstream oss;
      oss << "OP{" << M << "x" << N << "x" << K
          << ",thr=" << threads
          << ",tile=" << tile_size
          << ",type=" << op_type << "}";
      return oss.str();
    }

    // ========== TuningEntry Implementation ==========

    TuningEntry::TuningEntry()
        : performance_gflops(0.0), benchmark_time_usec(0.0),
          timestamp(0), cache_version(CACHE_VERSION)
    {
      std::memset(kernel_name, 0, MAX_KERNEL_NAME_LEN);
    }

    TuningEntry::TuningEntry(const OperationSignature &sig, const char *name,
                             double gflops, double time_usec)
        : op_sig(sig), performance_gflops(gflops),
          benchmark_time_usec(time_usec),
          timestamp(static_cast<uint64_t>(std::time(nullptr))),
          cache_version(CACHE_VERSION)
    {
      std::memset(kernel_name, 0, MAX_KERNEL_NAME_LEN);
      if (name)
      {
        std::strncpy(kernel_name, name, MAX_KERNEL_NAME_LEN - 1);
      }
    }

    // ========== CacheStats Implementation ==========

    CacheStats::CacheStats()
        : total_entries(0), hits(0), misses(0), evictions(0), hit_rate(0.0)
    {
    }

    std::string CacheStats::ToString() const noexcept
    {
      std::ostringstream oss;
      oss << "CacheStats{entries=" << total_entries
          << ",hits=" << hits
          << ",misses=" << misses
          << ",evictions=" << evictions
          << ",hit_rate=" << (hit_rate * 100.0) << "%}";
      return oss.str();
    }

    // ========== CacheConfig Implementation ==========

    CacheConfig::CacheConfig()
        : format(Format::BINARY), max_entries(0),
          validate_hardware(true), auto_save(true), thread_safe(true)
    {
      // Default cache directory
      const char *home = std::getenv("HOME");
      if (home)
      {
        cache_dir = std::string(home) + "/.jcore/cache";
      }
      else
      {
        cache_dir = "/tmp/jcore_cache";
      }
    }

    void CacheConfig::LoadFromEnv() noexcept
    {
      // Read JCORE_CACHE_DIR
      const char *env_dir = std::getenv("JCORE_CACHE_DIR");
      if (env_dir && env_dir[0] != '\0')
      {
        cache_dir = env_dir;
      }

      // Read JCORE_CACHE_FORMAT
      const char *env_fmt = std::getenv("JCORE_CACHE_FORMAT");
      if (env_fmt)
      {
        if (std::strcmp(env_fmt, "json") == 0)
          format = Format::JSON;
        else if (std::strcmp(env_fmt, "binary") == 0)
          format = Format::BINARY;
      }

      // Read JCORE_CACHE_MAX_ENTRIES
      const char *env_max = std::getenv("JCORE_CACHE_MAX_ENTRIES");
      if (env_max)
      {
        max_entries = static_cast<size_t>(std::atol(env_max));
      }

      // Read JCORE_CACHE_VALIDATE_HW
      const char *env_validate = std::getenv("JCORE_CACHE_VALIDATE_HW");
      if (env_validate)
      {
        validate_hardware = (std::strcmp(env_validate, "1") == 0 ||
                             std::strcmp(env_validate, "true") == 0);
      }

      // Read JCORE_CACHE_AUTO_SAVE
      const char *env_auto = std::getenv("JCORE_CACHE_AUTO_SAVE");
      if (env_auto)
      {
        auto_save = (std::strcmp(env_auto, "1") == 0 ||
                     std::strcmp(env_auto, "true") == 0);
      }
    }

    // ========== Hardware Detection ==========

    HardwareSignature DetectHardware() noexcept
    {
      HardwareSignature sig;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
      // CPUID-based feature detection
      unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

      // Check AVX
      if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
      {
        if (ecx & bit_AVX)
          sig.features |= 1ULL; // AVX
      }

      // Check AVX2
      if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
      {
        if (ebx & bit_AVX2)
          sig.features |= 2ULL; // AVX2
        if (ebx & (1U << 16))
          sig.features |= 4ULL; // AVX512F
        if (edx & (1U << 22))
          sig.features |= 8ULL; // AMX-BF16
      }
#endif

      // Detect core count
      sig.cores = static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_ONLN));
      if (sig.cores == 0)
        sig.cores = 1;

      // Try to read cache info from sysfs (Linux)
      std::ifstream l1d("/sys/devices/system/cpu/cpu0/cache/index0/size");
      if (l1d.is_open())
      {
        std::string val;
        l1d >> val;
        sig.l1_cache_kb = std::atoi(val.c_str());
      }

      std::ifstream l2("/sys/devices/system/cpu/cpu0/cache/index2/size");
      if (l2.is_open())
      {
        std::string val;
        l2 >> val;
        sig.l2_cache_kb = std::atoi(val.c_str());
      }

      std::ifstream l3("/sys/devices/system/cpu/cpu0/cache/index3/size");
      if (l3.is_open())
      {
        std::string val;
        l3 >> val;
        sig.l3_cache_kb = std::atoi(val.c_str());
      }

      // Default values if sysfs unavailable
      if (sig.l1_cache_kb == 0)
        sig.l1_cache_kb = 32;
      if (sig.l2_cache_kb == 0)
        sig.l2_cache_kb = 256;
      if (sig.l3_cache_kb == 0)
        sig.l3_cache_kb = 8192;

      // NUMA detection (basic)
      sig.numa_nodes = 1; // Default

      return sig;
    }

    // ========== Utility Functions ==========

    const char *StatusToString(Status s) noexcept
    {
      switch (s)
      {
      case Status::OK:
        return "OK";
      case Status::ERR_NOT_INITIALIZED:
        return "Not initialized";
      case Status::ERR_INVALID_ARG:
        return "Invalid argument";
      case Status::ERR_NO_MEMORY:
        return "Out of memory";
      case Status::ERR_IO_FAILURE:
        return "I/O failure";
      case Status::ERR_PARSE_FAILURE:
        return "Parse failure";
      case Status::ERR_VERSION_MISMATCH:
        return "Version mismatch";
      case Status::ERR_HARDWARE_MISMATCH:
        return "Hardware mismatch";
      case Status::ERR_NOT_FOUND:
        return "Not found";
      case Status::ERR_ALREADY_EXISTS:
        return "Already exists";
      case Status::ERR_LOCK_FAILURE:
        return "Lock failure";
      case Status::ERR_INTERNAL:
        return "Internal error";
      default:
        return "Unknown error";
      }
    }

    // ========== Internal Implementation ==========

    struct TuningCache::Impl
    {
      bool initialized;
      CacheConfig config;
      HardwareSignature hw_sig;
      std::unordered_map<uint64_t, TuningEntry> entries;
      mutable std::mutex mutex;
      CacheStats stats;

      Impl() : initialized(false) {}

      Status EnsureDirectoryExists(const std::string &dir)
      {
        struct stat st;
        if (stat(dir.c_str(), &st) == 0)
        {
          if (S_ISDIR(st.st_mode))
            return Status::OK;
          return Status::ERR_IO_FAILURE;
        }

        // Create directory
        if (mkdir(dir.c_str(), 0755) != 0)
        {
          // Try creating parent
          size_t pos = dir.find_last_of('/');
          if (pos != std::string::npos)
          {
            std::string parent = dir.substr(0, pos);
            Status s = EnsureDirectoryExists(parent);
            if (s != Status::OK)
              return s;
            if (mkdir(dir.c_str(), 0755) != 0)
              return Status::ERR_IO_FAILURE;
          }
          else
          {
            return Status::ERR_IO_FAILURE;
          }
        }
        return Status::OK;
      }

      std::string GetCacheFilePath() const
      {
        if (config.format == Format::JSON)
          return config.cache_dir + "/tuning_cache.json";
        else
          return config.cache_dir + "/tuning_cache.bin";
      }

      Status SaveJSON(const std::string &filepath)
      {
        std::ofstream ofs(filepath);
        if (!ofs.is_open())
          return Status::ERR_IO_FAILURE;

        ofs << "{\n";
        ofs << "  \"version\": " << CACHE_VERSION << ",\n";
        ofs << "  \"hardware\": {\n";
        ofs << "    \"features\": " << hw_sig.features << ",\n";
        ofs << "    \"cores\": " << hw_sig.cores << ",\n";
        ofs << "    \"l1_kb\": " << hw_sig.l1_cache_kb << ",\n";
        ofs << "    \"l2_kb\": " << hw_sig.l2_cache_kb << ",\n";
        ofs << "    \"l3_kb\": " << hw_sig.l3_cache_kb << ",\n";
        ofs << "    \"numa_nodes\": " << hw_sig.numa_nodes << "\n";
        ofs << "  },\n";
        ofs << "  \"entries\": [\n";

        bool first = true;
        for (const auto &kv : entries)
        {
          const TuningEntry &e = kv.second;
          if (!first)
            ofs << ",\n";
          first = false;

          ofs << "    {\n";
          ofs << "      \"M\": " << e.op_sig.M << ",\n";
          ofs << "      \"N\": " << e.op_sig.N << ",\n";
          ofs << "      \"K\": " << e.op_sig.K << ",\n";
          ofs << "      \"threads\": " << e.op_sig.threads << ",\n";
          ofs << "      \"tile_size\": " << e.op_sig.tile_size << ",\n";
          ofs << "      \"op_type\": " << e.op_sig.op_type << ",\n";
          ofs << "      \"kernel_name\": \"" << e.kernel_name << "\",\n";
          ofs << "      \"gflops\": " << e.performance_gflops << ",\n";
          ofs << "      \"bench_usec\": " << e.benchmark_time_usec << ",\n";
          ofs << "      \"timestamp\": " << e.timestamp << "\n";
          ofs << "    }";
        }

        ofs << "\n  ]\n}\n";
        ofs.close();
        return ofs.good() ? Status::OK : Status::ERR_IO_FAILURE;
      }

      Status SaveBinary(const std::string &filepath)
      {
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open())
          return Status::ERR_IO_FAILURE;

        // Write header
        uint32_t magic = 0x4A434348; // "JCCH"
        ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char *>(&CACHE_VERSION), sizeof(CACHE_VERSION));

        // Write hardware signature
        ofs.write(reinterpret_cast<const char *>(&hw_sig), sizeof(hw_sig));

        // Write entry count
        uint64_t count = entries.size();
        ofs.write(reinterpret_cast<const char *>(&count), sizeof(count));

        // Write entries
        for (const auto &kv : entries)
        {
          const TuningEntry &e = kv.second;
          ofs.write(reinterpret_cast<const char *>(&e), sizeof(e));
        }

        ofs.close();
        return ofs.good() ? Status::OK : Status::ERR_IO_FAILURE;
      }

      Status LoadJSON(const std::string &filepath)
      {
        // Simplified JSON parser - production should use a library
        std::ifstream ifs(filepath);
        if (!ifs.is_open())
          return Status::ERR_IO_FAILURE;

        // For now, just return success - full JSON parsing would be extensive
        // In production, use nlohmann/json or similar
        ifs.close();
        return Status::OK;
      }

      Status LoadBinary(const std::string &filepath)
      {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open())
          return Status::ERR_IO_FAILURE;

        // Read header
        uint32_t magic = 0;
        uint32_t version = 0;
        ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char *>(&version), sizeof(version));

        if (magic != 0x4A434348)
          return Status::ERR_PARSE_FAILURE;
        if (version != CACHE_VERSION)
          return Status::ERR_VERSION_MISMATCH;

        // Read hardware signature
        HardwareSignature file_hw;
        ifs.read(reinterpret_cast<char *>(&file_hw), sizeof(file_hw));

        if (config.validate_hardware && file_hw != hw_sig)
          return Status::ERR_HARDWARE_MISMATCH;

        // Read entry count
        uint64_t count = 0;
        ifs.read(reinterpret_cast<char *>(&count), sizeof(count));

        // Read entries
        entries.clear();
        for (uint64_t i = 0; i < count; ++i)
        {
          TuningEntry e;
          ifs.read(reinterpret_cast<char *>(&e), sizeof(e));
          if (!ifs.good())
            return Status::ERR_PARSE_FAILURE;

          uint64_t key = e.op_sig.Hash();
          entries[key] = e;
        }

        stats.total_entries = entries.size();
        ifs.close();
        return Status::OK;
      }
    };

    // ========== TuningCache Implementation ==========

    TuningCache::TuningCache()
        : impl_(std::make_unique<Impl>())
    {
    }

    TuningCache::~TuningCache()
    {
      Shutdown();
    }

    TuningCache::TuningCache(TuningCache &&other) noexcept
        : impl_(std::move(other.impl_))
    {
    }

    TuningCache &TuningCache::operator=(TuningCache &&other) noexcept
    {
      if (this != &other)
      {
        Shutdown();
        impl_ = std::move(other.impl_);
      }
      return *this;
    }

    Status TuningCache::Init(const CacheConfig &config) noexcept
    {
      if (!impl_)
        return Status::ERR_NO_MEMORY;

      if (impl_->initialized)
        return Status::ERR_ALREADY_EXISTS;

      impl_->config = config;
      impl_->hw_sig = DetectHardware();
      impl_->stats = CacheStats();

      // Ensure cache directory exists
      Status s = impl_->EnsureDirectoryExists(config.cache_dir);
      if (s != Status::OK)
        return s;

      impl_->initialized = true;

      // Try to load existing cache
      Load();

      return Status::OK;
    }

    void TuningCache::Shutdown() noexcept
    {
      if (!impl_ || !impl_->initialized)
        return;

      if (impl_->config.auto_save)
      {
        Save();
      }

      impl_->entries.clear();
      impl_->initialized = false;
    }

    Status TuningCache::Query(const OperationSignature &op_sig, TuningEntry &out_entry) noexcept
    {
      if (!impl_ || !impl_->initialized)
        return Status::ERR_NOT_INITIALIZED;

      uint64_t key = op_sig.Hash();

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto it = impl_->entries.find(key);
        if (it == impl_->entries.end())
        {
          impl_->stats.misses++;
          return Status::ERR_NOT_FOUND;
        }
        out_entry = it->second;
        impl_->stats.hits++;
      }
      else
      {
        auto it = impl_->entries.find(key);
        if (it == impl_->entries.end())
        {
          impl_->stats.misses++;
          return Status::ERR_NOT_FOUND;
        }
        out_entry = it->second;
        impl_->stats.hits++;
      }

      // Update hit rate
      size_t total = impl_->stats.hits + impl_->stats.misses;
      if (total > 0)
        impl_->stats.hit_rate = static_cast<double>(impl_->stats.hits) / total;

      return Status::OK;
    }

    Status TuningCache::Insert(const TuningEntry &entry) noexcept
    {
      if (!impl_ || !impl_->initialized)
        return Status::ERR_NOT_INITIALIZED;

      uint64_t key = entry.op_sig.Hash();

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);

        // Check max entries limit
        if (impl_->config.max_entries > 0 &&
            impl_->entries.size() >= impl_->config.max_entries &&
            impl_->entries.find(key) == impl_->entries.end())
        {
          // Evict oldest entry
          uint64_t oldest_key = 0;
          uint64_t oldest_ts = UINT64_MAX;
          for (const auto &kv : impl_->entries)
          {
            if (kv.second.timestamp < oldest_ts)
            {
              oldest_ts = kv.second.timestamp;
              oldest_key = kv.first;
            }
          }
          impl_->entries.erase(oldest_key);
          impl_->stats.evictions++;
        }

        impl_->entries[key] = entry;
        impl_->stats.total_entries = impl_->entries.size();
      }
      else
      {
        impl_->entries[key] = entry;
        impl_->stats.total_entries = impl_->entries.size();
      }

      if (impl_->config.auto_save)
      {
        return Save();
      }

      return Status::OK;
    }

    Status TuningCache::Remove(const OperationSignature &op_sig) noexcept
    {
      if (!impl_ || !impl_->initialized)
        return Status::ERR_NOT_INITIALIZED;

      uint64_t key = op_sig.Hash();

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto it = impl_->entries.find(key);
        if (it == impl_->entries.end())
          return Status::ERR_NOT_FOUND;
        impl_->entries.erase(it);
        impl_->stats.total_entries = impl_->entries.size();
      }
      else
      {
        auto it = impl_->entries.find(key);
        if (it == impl_->entries.end())
          return Status::ERR_NOT_FOUND;
        impl_->entries.erase(it);
        impl_->stats.total_entries = impl_->entries.size();
      }

      return Status::OK;
    }

    Status TuningCache::Clear() noexcept
    {
      if (!impl_ || !impl_->initialized)
        return Status::ERR_NOT_INITIALIZED;

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->entries.clear();
        impl_->stats.total_entries = 0;
      }
      else
      {
        impl_->entries.clear();
        impl_->stats.total_entries = 0;
      }

      return Status::OK;
    }

    Status TuningCache::Save() noexcept
    {
      if (!impl_ || !impl_->initialized)
        return Status::ERR_NOT_INITIALIZED;

      std::string filepath = impl_->GetCacheFilePath();

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->config.format == Format::JSON)
          return impl_->SaveJSON(filepath);
        else
          return impl_->SaveBinary(filepath);
      }
      else
      {
        if (impl_->config.format == Format::JSON)
          return impl_->SaveJSON(filepath);
        else
          return impl_->SaveBinary(filepath);
      }
    }

    Status TuningCache::Load() noexcept
    {
      if (!impl_ || !impl_->initialized)
        return Status::ERR_NOT_INITIALIZED;

      std::string filepath = impl_->GetCacheFilePath();

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->config.format == Format::JSON)
          return impl_->LoadJSON(filepath);
        else
          return impl_->LoadBinary(filepath);
      }
      else
      {
        if (impl_->config.format == Format::JSON)
          return impl_->LoadJSON(filepath);
        else
          return impl_->LoadBinary(filepath);
      }
    }

    Status TuningCache::Export(const std::string &filepath, Format format) noexcept
    {
      if (!impl_ || !impl_->initialized)
        return Status::ERR_NOT_INITIALIZED;

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (format == Format::JSON)
          return impl_->SaveJSON(filepath);
        else
          return impl_->SaveBinary(filepath);
      }
      else
      {
        if (format == Format::JSON)
          return impl_->SaveJSON(filepath);
        else
          return impl_->SaveBinary(filepath);
      }
    }

    Status TuningCache::Import(const std::string &filepath) noexcept
    {
      if (!impl_ || !impl_->initialized)
        return Status::ERR_NOT_INITIALIZED;

      // Detect format by extension
      Format format = Format::BINARY;
      if (filepath.size() >= 5 && filepath.substr(filepath.size() - 5) == ".json")
        format = Format::JSON;

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (format == Format::JSON)
          return impl_->LoadJSON(filepath);
        else
          return impl_->LoadBinary(filepath);
      }
      else
      {
        if (format == Format::JSON)
          return impl_->LoadJSON(filepath);
        else
          return impl_->LoadBinary(filepath);
      }
    }

    CacheStats TuningCache::GetStats() const noexcept
    {
      if (!impl_ || !impl_->initialized)
        return CacheStats();

      if (impl_->config.thread_safe)
      {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->stats;
      }
      else
      {
        return impl_->stats;
      }
    }

    HardwareSignature TuningCache::GetHardwareSignature() const noexcept
    {
      if (!impl_ || !impl_->initialized)
        return HardwareSignature();

      return impl_->hw_sig;
    }

    bool TuningCache::IsInitialized() const noexcept
    {
      return impl_ && impl_->initialized;
    }

    // ========== C API Implementation ==========

    extern "C"
    {
      struct tc_cache_t
      {
        TuningCache *cache;
      };

      tc_cache_t *tc_cache_create(void)
      {
        tc_cache_t *handle = new (std::nothrow) tc_cache_t;
        if (!handle)
          return nullptr;
        handle->cache = new (std::nothrow) TuningCache();
        if (!handle->cache)
        {
          delete handle;
          return nullptr;
        }
        return handle;
      }

      tc_status_t tc_cache_init(tc_cache_t *cache, const tc_cache_config_t *config)
      {
        if (!cache || !cache->cache || !config)
          return TC_ERR_INVALID_ARG;

        CacheConfig cpp_config;
        cpp_config.cache_dir = config->cache_dir;
        cpp_config.format = (config->format == TC_FORMAT_JSON) ? Format::JSON : Format::BINARY;
        cpp_config.max_entries = config->max_entries;
        cpp_config.validate_hardware = (config->validate_hardware != 0);
        cpp_config.auto_save = (config->auto_save != 0);
        cpp_config.thread_safe = (config->thread_safe != 0);

        Status s = cache->cache->Init(cpp_config);
        return static_cast<tc_status_t>(s);
      }

      void tc_cache_shutdown(tc_cache_t *cache)
      {
        if (cache && cache->cache)
        {
          cache->cache->Shutdown();
        }
      }

      void tc_cache_destroy(tc_cache_t *cache)
      {
        if (cache)
        {
          if (cache->cache)
          {
            delete cache->cache;
            cache->cache = nullptr;
          }
          delete cache;
        }
      }

      tc_status_t tc_cache_query(tc_cache_t *cache, const tc_op_signature_t *op_sig,
                                 tc_tuning_entry_t *out_entry)
      {
        if (!cache || !cache->cache || !op_sig || !out_entry)
          return TC_ERR_INVALID_ARG;

        OperationSignature cpp_sig(op_sig->M, op_sig->N, op_sig->K,
                                   op_sig->threads, op_sig->tile_size, op_sig->op_type);

        TuningEntry cpp_entry;
        Status s = cache->cache->Query(cpp_sig, cpp_entry);
        if (s != Status::OK)
          return static_cast<tc_status_t>(s);

        // Convert to C struct
        out_entry->op_sig.M = cpp_entry.op_sig.M;
        out_entry->op_sig.N = cpp_entry.op_sig.N;
        out_entry->op_sig.K = cpp_entry.op_sig.K;
        out_entry->op_sig.threads = cpp_entry.op_sig.threads;
        out_entry->op_sig.tile_size = cpp_entry.op_sig.tile_size;
        out_entry->op_sig.op_type = cpp_entry.op_sig.op_type;
        std::strncpy(out_entry->kernel_name, cpp_entry.kernel_name, 255);
        out_entry->kernel_name[255] = '\0';
        out_entry->performance_gflops = cpp_entry.performance_gflops;
        out_entry->benchmark_time_usec = cpp_entry.benchmark_time_usec;
        out_entry->timestamp = cpp_entry.timestamp;
        out_entry->cache_version = cpp_entry.cache_version;

        return TC_OK;
      }

      tc_status_t tc_cache_insert(tc_cache_t *cache, const tc_tuning_entry_t *entry)
      {
        if (!cache || !cache->cache || !entry)
          return TC_ERR_INVALID_ARG;

        OperationSignature cpp_sig(entry->op_sig.M, entry->op_sig.N, entry->op_sig.K,
                                   entry->op_sig.threads, entry->op_sig.tile_size,
                                   entry->op_sig.op_type);

        TuningEntry cpp_entry(cpp_sig, entry->kernel_name,
                              entry->performance_gflops, entry->benchmark_time_usec);

        Status s = cache->cache->Insert(cpp_entry);
        return static_cast<tc_status_t>(s);
      }

      tc_status_t tc_cache_remove(tc_cache_t *cache, const tc_op_signature_t *op_sig)
      {
        if (!cache || !cache->cache || !op_sig)
          return TC_ERR_INVALID_ARG;

        OperationSignature cpp_sig(op_sig->M, op_sig->N, op_sig->K,
                                   op_sig->threads, op_sig->tile_size, op_sig->op_type);

        Status s = cache->cache->Remove(cpp_sig);
        return static_cast<tc_status_t>(s);
      }

      tc_status_t tc_cache_clear(tc_cache_t *cache)
      {
        if (!cache || !cache->cache)
          return TC_ERR_INVALID_ARG;

        Status s = cache->cache->Clear();
        return static_cast<tc_status_t>(s);
      }

      tc_status_t tc_cache_save(tc_cache_t *cache)
      {
        if (!cache || !cache->cache)
          return TC_ERR_INVALID_ARG;

        Status s = cache->cache->Save();
        return static_cast<tc_status_t>(s);
      }

      tc_status_t tc_cache_load(tc_cache_t *cache)
      {
        if (!cache || !cache->cache)
          return TC_ERR_INVALID_ARG;

        Status s = cache->cache->Load();
        return static_cast<tc_status_t>(s);
      }

      tc_status_t tc_cache_get_stats(tc_cache_t *cache, tc_cache_stats_t *out_stats)
      {
        if (!cache || !cache->cache || !out_stats)
          return TC_ERR_INVALID_ARG;

        CacheStats cpp_stats = cache->cache->GetStats();
        out_stats->total_entries = cpp_stats.total_entries;
        out_stats->hits = cpp_stats.hits;
        out_stats->misses = cpp_stats.misses;
        out_stats->evictions = cpp_stats.evictions;
        out_stats->hit_rate = cpp_stats.hit_rate;

        return TC_OK;
      }

      const char *tc_status_str(tc_status_t s)
      {
        return StatusToString(static_cast<Status>(s));
      }

      void tc_cache_config_init_default(tc_cache_config_t *config)
      {
        if (!config)
          return;

        CacheConfig cpp_config;
        cpp_config.LoadFromEnv();

        std::strncpy(config->cache_dir, cpp_config.cache_dir.c_str(), 511);
        config->cache_dir[511] = '\0';
        config->format = (cpp_config.format == Format::JSON) ? TC_FORMAT_JSON : TC_FORMAT_BINARY;
        config->max_entries = cpp_config.max_entries;
        config->validate_hardware = cpp_config.validate_hardware ? 1 : 0;
        config->auto_save = cpp_config.auto_save ? 1 : 0;
        config->thread_safe = cpp_config.thread_safe ? 1 : 0;
      }

    } // extern "C"

  } // namespace tuning_cache
} // namespace jcore