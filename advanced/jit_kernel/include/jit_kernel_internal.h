/// advanced/jit_kernel/include/jit_kernel_internal.h
#ifndef JIT_KERNEL_INTERNAL_H_
#define JIT_KERNEL_INTERNAL_H_

#include "jit_kernel_generator.h"

// LLVM headers

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

// Kernel Fusion Engine headers
#include "kernel_fusion_engine_internal.h"


// Base component headers
#include "benchmark.h"
#include "cache_info.h"
#include "cpu_features.h"
#include "cpu_info.h"

// Standard C++ headers
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

    // LLVM 20 forward declarations (clean API, stable types)
  namespace llvm {
  class LLVMContext;
  class Module;
  class Function;
  class Value;
  class Type;
  class ConstantFolder;
  class IRBuilderDefaultInserter;
  template <typename FolderTy, typename InserterTy> class IRBuilder;
  class TargetMachine;

  namespace orc {
  class LLJIT;
  class ThreadSafeContext;
  class ThreadSafeModule;
  class ExecutionSession;
  class JITDylib;
  } // namespace orc
} // namespace llvm

// Use standard IRBuilder type for LLVM 20
using LLVMIRBuilder =
    llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>;

// Vectorization library detection
#ifdef __x86_64__
#define JKG_HAS_X86 1
#else
#define JKG_HAS_X86 0
#endif

// Highway support
#ifdef HWY_HIGHWAY_H_
#define JKG_HAS_HIGHWAY 1
#else
#define JKG_HAS_HIGHWAY 0
#endif

// VectorClass support
#ifdef VECTORCLASS_H
#define JKG_HAS_VECTORCLASS 1
#else
#define JKG_HAS_VECTORCLASS 0
#endif

// EVE support
#ifdef EVE_MODULE_CORE_HPP_INCLUDED
#define JKG_HAS_EVE 1
#else
#define JKG_HAS_EVE 0
#endif

namespace jkg_internal {

/* ========================================================================== */
/* Constants */
/* ========================================================================== */

constexpr size_t DEFAULT_CACHE_SIZE = 256; // Maximum cached kernels
constexpr size_t MAX_KERNEL_NAME_LEN = 128;
constexpr int DEFAULT_OPTIMIZATION_LEVEL = 2;

/* ========================================================================== */
/* Internal Kernel Handle Structure (inside namespace)                        */
/* ========================================================================== */

struct jkg_kernel_impl_t {
  void *function_ptr;         // Compiled function pointer
  jkg_kernel_type_t type;     // Kernel type
  jkg_kernel_params_t params; // Generation parameters
  std::string llvm_ir;        // LLVM IR (for debugging)
  std::string assembly;       // Native assembly (for debugging)
  std::string kernel_name;    // Unique kernel identifier
  size_t code_size_bytes;     // Size of generated code
  bool is_cached;             // Loaded from cache?
  std::atomic<int> ref_count; // Reference count

  jkg_kernel_impl_t()
      : function_ptr(nullptr), type(JKG_KERNEL_GEMM_TILE), code_size_bytes(0),
        is_cached(false), ref_count(1) {}
};

// Cast helpers for opaque pointer conversion
inline jkg_kernel_impl_t *handle_to_impl(jkg_kernel_internal_t *handle) {
  return reinterpret_cast<jkg_kernel_impl_t *>(handle);
}

inline jkg_kernel_internal_t *impl_to_handle(jkg_kernel_impl_t *impl) {
  return reinterpret_cast<jkg_kernel_internal_t *>(static_cast<void *>(impl));
}

/* ========================================================================== */
/* Kernel Cache Key */
/* ========================================================================== */

struct KernelCacheKey {
  jkg_kernel_type_t type;
  jkg_isa_t isa;
  size_t M, N, K;
  jkg_activation_t activation;
  int has_bias;
  int has_residual;

  bool operator==(const KernelCacheKey &other) const {
    return type == other.type && isa == other.isa && M == other.M &&
           N == other.N && K == other.K && activation == other.activation &&
           has_bias == other.has_bias && has_residual == other.has_residual;
  }
};

// Hash function for cache key
struct KernelCacheKeyHash {
  size_t operator()(const KernelCacheKey &k) const {
    size_t h = std::hash<int>{}(static_cast<int>(k.type));
    h ^= std::hash<int>{}(static_cast<int>(k.isa)) << 1;
    h ^= std::hash<size_t>{}(k.M) << 2;
    h ^= std::hash<size_t>{}(k.N) << 3;
    h ^= std::hash<size_t>{}(k.K) << 4;
    h ^= std::hash<int>{}(static_cast<int>(k.activation)) << 5;
    h ^= std::hash<int>{}(k.has_bias) << 6;
    h ^= std::hash<int>{}(k.has_residual) << 7;
    return h;
  }
};

/* ========================================================================== */
/* Compiled Kernel Handle (Internal)                                          */
/* ========================================================================== */

struct jkg_kernel_t {
  void *function_ptr;         // Compiled function pointer
  jkg_kernel_type_t type;     // Kernel type
  jkg_kernel_params_t params; // Generation parameters
  std::string llvm_ir;        // LLVM IR (for debugging)
  std::string assembly;       // Native assembly (for debugging)
  std::string kernel_name;    // Unique kernel identifier
  size_t code_size_bytes;     // Size of generated code
  bool is_cached;             // Loaded from cache?
  std::atomic<int> ref_count; // Reference count

  jkg_kernel_t()
      : function_ptr(nullptr), type(JKG_KERNEL_GEMM_TILE), code_size_bytes(0),
        is_cached(false), ref_count(1) {}
};

/* ========================================================================== */
/* JIT Compiler State */
/* ========================================================================== */

struct JKGState {
  bool initialized;
  jkg_config_t config;

  // LLVM infrastructure
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unique_ptr<llvm::orc::ThreadSafeContext> ts_context;
  llvm::TargetMachine *target_machine;

  // CPU capabilities
  CPUFeatures cpu_features;
  cpu_info_t cpu_info;
  std::vector<CacheInfo> cache_info;
  uint32_t available_isa_mask;

  // Kernel cache
  std::unordered_map<KernelCacheKey, std::shared_ptr<jkg_kernel_impl_t>,
                     KernelCacheKeyHash>
      kernel_cache;
  std::mutex cache_mutex;
  std::atomic<size_t> cache_hits{0};
  std::atomic<size_t> cache_misses{0};

  // Statistics
  std::atomic<size_t> kernels_generated{0};
  std::atomic<size_t> total_compile_time_us{0};

  JKGState()
      : initialized(false), target_machine(nullptr), available_isa_mask(0) {}
};

// Global state instance (defined in jit_core.cpp)
extern JKGState g_jkg_state;

/* ========================================================================== */
/* LLVM IR Generation Classes */
/* ========================================================================== */

class IRGenerator {
public:
  virtual ~IRGenerator() = default;

  virtual llvm::Function *generate(llvm::Module *module, LLVMIRBuilder *builder,
                                   const jkg_kernel_params_t &params) = 0;
};

class GEMMTileGenerator : public IRGenerator {
public:
  llvm::Function *generate(llvm::Module *module, LLVMIRBuilder *builder,
                           const jkg_kernel_params_t &params) override;

protected:
  void emit_gemm_loop(LLVMIRBuilder *builder, llvm::Function *func, size_t M,
                      size_t N, size_t K, jkg_isa_t isa);
};

class FusedGEMMGenerator : public GEMMTileGenerator {  // Inherit from GEMMTileGenerator
public:
    llvm::Function *generate(llvm::Module *module, LLVMIRBuilder *builder,
                             const jkg_kernel_params_t &params) override;
private:
    llvm::Value *emit_epilogue(LLVMIRBuilder *builder, llvm::Value *result,
                               const jkg_kernel_params_t &params);
};

class ElementwiseGenerator : public IRGenerator {
public:
  llvm::Function *generate(llvm::Module *module, LLVMIRBuilder *builder,
                           const jkg_kernel_params_t &params) override;
};

/* ========================================================================== */
/* Vectorization Backend Interfaces */
/* ========================================================================== */

class VectorizationBackend {
public:
  virtual ~VectorizationBackend() = default;

  virtual bool is_available() const = 0;
  virtual size_t vector_width() const = 0;
  virtual std::string name() const = 0;

  // Generate vectorized operations
  virtual void generate_gemm_tile(llvm::Module *module, LLVMIRBuilder *builder,
                                  llvm::Function *func, size_t M, size_t N,
                                  size_t K) = 0;

  virtual void generate_activation(llvm::Module *module, LLVMIRBuilder *builder,
                                   llvm::Value *data, size_t N,
                                   jkg_activation_t act) = 0;
};

class HighwayBackend : public VectorizationBackend {
public:
  bool is_available() const override;
  size_t vector_width() const override;
  std::string name() const override { return "Highway"; }

  void generate_gemm_tile(llvm::Module *module, LLVMIRBuilder *builder,
                          llvm::Function *func, size_t M, size_t N,
                          size_t K) override;
  void generate_activation(llvm::Module *module, LLVMIRBuilder *builder,
                           llvm::Value *data, size_t N,
                           jkg_activation_t act) override;
};

class VectorClassBackend : public VectorizationBackend {
public:
  bool is_available() const override;
  size_t vector_width() const override;
  std::string name() const override { return "VectorClass"; }

  void generate_gemm_tile(llvm::Module *module, LLVMIRBuilder *builder,
                          llvm::Function *func, size_t M, size_t N,
                          size_t K) override;
  void generate_activation(llvm::Module *module, LLVMIRBuilder *builder,
                           llvm::Value *data, size_t N,
                           jkg_activation_t act) override;
};

class EVEBackend : public VectorizationBackend {
public:
  bool is_available() const override;
  size_t vector_width() const override;
  std::string name() const override { return "EVE"; }

  void generate_gemm_tile(llvm::Module *module, LLVMIRBuilder *builder,
                          llvm::Function *func, size_t M, size_t N,
                          size_t K) override;
  void generate_activation(llvm::Module *module, LLVMIRBuilder *builder,
                           llvm::Value *data, size_t N,
                           jkg_activation_t act) override;
};

/* ========================================================================== */
/* Helper Functions */
/* ========================================================================== */

// Reset LLVM Initialization
void reset_llvm_init();

// ISA detection and conversion
uint32_t detect_available_isa();
jkg_isa_t select_best_isa(uint32_t available_mask);
jkg_isa_t convert_cpu_features_to_isa(const CPUFeatures &features);

// Cache management
std::shared_ptr<jkg_kernel_impl_t>
lookup_cached_kernel(const KernelCacheKey &key);
void insert_cached_kernel(const KernelCacheKey &key,
                          std::shared_ptr<jkg_kernel_impl_t> handle);

// LLVM utilities
std::string get_target_triple();
std::string get_target_cpu();
std::string get_target_features(jkg_isa_t isa);
llvm::Function *create_function_stub(llvm::Module *module,
                                     jkg_kernel_type_t type,
                                     const jkg_kernel_params_t &params);

// Tile size optimization
void compute_optimal_tile_sizes(jkg_isa_t isa,
                                const std::vector<CacheInfo> &cache,
                                size_t &out_M, size_t &out_N, size_t &out_K);

// Kernel name generation
std::string generate_kernel_name(jkg_kernel_type_t type,
                                 const jkg_kernel_params_t &params,
                                 jkg_isa_t isa);

// Backend selection
std::unique_ptr<VectorizationBackend> create_backend(jkg_backend_t backend);
jkg_backend_t select_best_backend(jkg_isa_t isa);

// Error handling
void log_error(const char *format, ...);
void log_info(const char *format, ...);

} // namespace jkg_internal

#endif /* JIT_KERNEL_INTERNAL_H_ */