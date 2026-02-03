// advanced/jit_kernel/src/llvm_codegen.cpp
#include "jit_kernel_internal.h"

// LLVM 20 includes - use system paths
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/GVN.h>

#include <chrono>

using namespace llvm;
using namespace llvm::orc;

namespace jkg_internal {

/* ========================================================================== */
/* LLVM Infrastructure Initialization (LLVM 20)                               */
/* ========================================================================== */

static bool llvm_initialized = false;
static int initialize_llvm() {

  if (llvm_initialized) {
    return JKG_OK;
  }

  log_info("Initializing LLVM 20...");

  // Initialize native target for JIT
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Create LLJIT instance (LLVM 20 stable API)
  auto jit_expected = llvm::orc::LLJITBuilder().create();
  if (!jit_expected) {
    std::string error_msg;
    llvm::handleAllErrors(
        jit_expected.takeError(),
        [&](const llvm::ErrorInfoBase &eib) { error_msg = eib.message(); });
    log_error("Failed to create LLJIT: %s", error_msg.c_str());
    return JKG_ERR_LLVM_INIT;
  }

  g_jkg_state.jit = std::move(*jit_expected);

  // Create thread-safe context (LLVM 20 ORC)
  g_jkg_state.ts_context = std::make_unique<llvm::orc::ThreadSafeContext>(
      std::make_unique<llvm::LLVMContext>());

  // Target machine is managed internally by LLJIT in LLVM 20
  g_jkg_state.target_machine = nullptr;

  llvm_initialized = true;

  // Declare KFE external functions for JIT linking
  auto ctx_lock = g_jkg_state.ts_context->getLock();
  auto &ctx = *g_jkg_state.ts_context->getContext();
  auto temp_module = std::make_unique<Module>("kfe_declarations", ctx);

  Type *floatTy = Type::getFloatTy(ctx);
  Type *int32Ty = Type::getInt32Ty(ctx);
  Type *int64Ty = Type::getInt64Ty(ctx);
  PointerType *floatPtrTy = PointerType::get(floatTy, 0);
  PointerType *voidPtrTy = PointerType::get(Type::getInt8Ty(ctx), 0);

  // Declare kfe_sgemm_bias_activation
  FunctionType *kfe_sgemm_bias_act_ty = FunctionType::get(
      int32Ty,
      {int32Ty, int32Ty, int32Ty, int64Ty, int64Ty, int64Ty, floatTy,
       floatPtrTy, int64Ty, floatPtrTy, int64Ty, floatPtrTy, int32Ty,
       floatPtrTy, int64Ty, voidPtrTy},
      false);
  Function::Create(kfe_sgemm_bias_act_ty, Function::ExternalLinkage,
                   "kfe_sgemm_bias_activation", temp_module.get());

  // Add KFE symbols to JIT
  auto &JD = g_jkg_state.jit->getMainJITDylib();
  llvm::orc::SymbolMap kfe_symbols;
  kfe_symbols[g_jkg_state.jit->mangleAndIntern("kfe_sgemm_bias_activation")] =
      llvm::orc::ExecutorSymbolDef(
          llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&kfe_sgemm_bias_activation)),
          llvm::JITSymbolFlags::Exported);

  if (auto err = JD.define(llvm::orc::absoluteSymbols(kfe_symbols))) {
    log_error("Failed to add KFE symbols: %s", toString(std::move(err)).c_str());
    return JKG_ERR_LLVM_INIT;
  }

  log_info("LLVM 20 initialized successfully");

  return JKG_OK;
}

void reset_llvm_init() {
  llvm_initialized = false;
}
/* ========================================================================== */
/* Function Stub Creation */
/* ========================================================================== */

Function *create_function_stub(Module *module, jkg_kernel_type_t type,
                               const jkg_kernel_params_t &params) {
  LLVMContext &ctx = module->getContext();

  // Common types
  Type *floatTy = Type::getFloatTy(ctx);
  PointerType *floatPtrTy = PointerType::get(floatTy, 0);
  Type *int64Ty = Type::getInt64Ty(ctx);
  Type *voidTy = Type::getVoidTy(ctx);

  std::vector<Type *> arg_types;
  std::string func_name =
      generate_kernel_name(type, params, g_jkg_state.config.target_isa);

  switch (type) {
  case JKG_KERNEL_GEMM_TILE:
  case JKG_KERNEL_GEMM_BIAS:
  case JKG_KERNEL_GEMM_BIAS_RELU:
  case JKG_KERNEL_GEMM_BIAS_ACT:
    // CRITICAL: Keep argument order consistent!
    // Signature: (A, B, C, M, N, K, lda, ldb, ldc, alpha, [bias], [beta])
    // Matrix pointers first
    arg_types.push_back(floatPtrTy); // A
    arg_types.push_back(floatPtrTy); // B
    arg_types.push_back(floatPtrTy); // C

    // Then dimensions (always i64)
    arg_types.push_back(int64Ty); // M
    arg_types.push_back(int64Ty); // N
    arg_types.push_back(int64Ty); // K
    arg_types.push_back(int64Ty); // lda
    arg_types.push_back(int64Ty); // ldb
    arg_types.push_back(int64Ty); // ldc

    // Then scalars
    arg_types.push_back(floatTy); // alpha

    // Optional arguments at the END
    if (params.has_bias) {
      arg_types.push_back(floatPtrTy); // bias
    }
    if (params.has_residual) {
      arg_types.push_back(floatTy); // beta
    }
    break;

  case JKG_KERNEL_ELEMENTWISE_ADD:
  case JKG_KERNEL_ELEMENTWISE_MUL:
    // Elementwise signature: (A, B, C, N)
    arg_types = {
        floatPtrTy, // A
        floatPtrTy, // B
        floatPtrTy, // C
        int64Ty     // N
    };
    break;

  case JKG_KERNEL_ACTIVATION:
    // Activation signature: (data, N)
    arg_types = {
        floatPtrTy, // data (in-place)
        int64Ty     // N
    };
    break;

  default:
    log_error("Unsupported kernel type: %d", type);
    return nullptr;
  }

  FunctionType *func_type = FunctionType::get(voidTy, arg_types, false);
  Function *func =
      Function::Create(func_type, Function::ExternalLinkage, func_name, module);

  // Set function attributes for LLVM 20
  func->addFnAttr(Attribute::AlwaysInline);
  func->addFnAttr(Attribute::NoUnwind);

  return func;
}

/* ========================================================================== */
/* GEMM Tile Generator Implementation                                         */
/* ========================================================================== */

Function *GEMMTileGenerator::generate(Module *module, LLVMIRBuilder *builder,
                                      const jkg_kernel_params_t &params) {
  Function *func = create_function_stub(module, JKG_KERNEL_GEMM_TILE, params);
  if (!func) {
    return nullptr;
  }

  // Create entry block
  BasicBlock *entry = BasicBlock::Create(module->getContext(), "entry", func);
  builder->SetInsertPoint(entry);

  // Extract arguments
  auto args = func->arg_begin();
  Value *A = &*args++;
  A->setName("A");
  Value *B = &*args++;
  B->setName("B");
  Value *C = &*args++;
  C->setName("C");
  Value *M = &*args++;
  M->setName("M");
  Value *N = &*args++;
  N->setName("N");
  Value *K = &*args++;
  K->setName("K");
  Value *lda = &*args++;
  lda->setName("lda");
  Value *ldb = &*args++;
  ldb->setName("ldb");
  Value *ldc = &*args++;
  ldc->setName("ldc");
  Value *alpha = &*args++;
  alpha->setName("alpha");

  // Emit the GEMM computation
  emit_gemm_loop(builder, func, params.M, params.N, params.K,
                 g_jkg_state.config.target_isa);
  builder->CreateRetVoid();

  // Verify function
  std::string error_str;
  llvm::raw_string_ostream error_stream(error_str);
  if (verifyFunction(*func, &error_stream)) {
    printf("Function verification failed for GEMM tile: %s", error_str.c_str());
    func->eraseFromParent();
    return nullptr;
  }

  return func;
}

void GEMMTileGenerator::emit_gemm_loop(LLVMIRBuilder *builder, Function *func,
                                       size_t M, size_t N, size_t K,
                                       jkg_isa_t isa) {
  LLVMContext &ctx = func->getContext();
  Module *module = func->getParent();

  // Get function arguments
  auto args = func->arg_begin();
  Value *A_ptr = &*args++;
  Value *B_ptr = &*args++;
  Value *C_ptr = &*args++;
  Value *M_arg = &*args++;
  Value *N_arg = &*args++;
  Value *K_arg = &*args++;
  Value *lda = &*args++;
  Value *ldb = &*args++;
  Value *ldc = &*args++;
  Value *alpha = &*args++;

  Type *int32Ty = Type::getInt32Ty(ctx);
  Type *int64Ty = Type::getInt64Ty(ctx);
  Type *floatTy = Type::getFloatTy(ctx);
  PointerType *floatPtrTy = PointerType::get(floatTy, 0);
  PointerType *voidPtrTy = PointerType::get(Type::getInt8Ty(ctx), 0);

  // Declare KFE function in this module
  FunctionType *kfe_sgemm_bias_act_ty = FunctionType::get(
      int32Ty,
      {int32Ty, int32Ty, int32Ty, int64Ty, int64Ty, int64Ty, floatTy,
       floatPtrTy, int64Ty, floatPtrTy, int64Ty, floatPtrTy, int32Ty,
       floatPtrTy, int64Ty, voidPtrTy},
      false);

  Function *kfe_func = module->getFunction("kfe_sgemm_bias_activation");
  if (!kfe_func) {
    kfe_func = Function::Create(kfe_sgemm_bias_act_ty, Function::ExternalLinkage,
                                 "kfe_sgemm_bias_activation", module);
  }

  // Prepare KFE arguments
  Value *layout = ConstantInt::get(int32Ty, 0); // KFE_LAYOUT_ROW_MAJOR
  Value *trans_a = ConstantInt::get(int32Ty, 0); // KFE_NO_TRANS
  Value *trans_b = ConstantInt::get(int32Ty, 0); // KFE_NO_TRANS
  Value *bias_null = ConstantPointerNull::get(floatPtrTy);
  Value *activation = ConstantInt::get(int32Ty, 0); // KFE_ACTIVATION_NONE
  Value *stats_null = ConstantPointerNull::get(voidPtrTy);

  // Cast dimensions to i64 if needed
  Value *M64 = builder->CreateIntCast(M_arg, int64Ty, false);
  Value *N64 = builder->CreateIntCast(N_arg, int64Ty, false);
  Value *K64 = builder->CreateIntCast(K_arg, int64Ty, false);
  Value *lda64 = builder->CreateIntCast(lda, int64Ty, false);
  Value *ldb64 = builder->CreateIntCast(ldb, int64Ty, false);
  Value *ldc64 = builder->CreateIntCast(ldc, int64Ty, false);

  // Call KFE: kfe_sgemm_bias_activation(layout, trans_a, trans_b, m, n, k, alpha,
  //                                      A, lda, B, ldb, bias, activation, C, ldc, stats)
  builder->CreateCall(kfe_func, {layout, trans_a, trans_b, M64, N64, K64, alpha,
                                  A_ptr, lda64, B_ptr, ldb64, bias_null, activation,
                                  C_ptr, ldc64, stats_null});
}

/* ========================================================================== */
/* Fused GEMM Generator Implementation                                        */
/* ========================================================================== */

Function *FusedGEMMGenerator::generate(Module *module, LLVMIRBuilder *builder,
                                       const jkg_kernel_params_t &params) {
  // Determine the actual kernel type
  jkg_kernel_type_t actual_type = JKG_KERNEL_GEMM_BIAS;
  if (params.activation == JKG_ACT_RELU) {
    actual_type = JKG_KERNEL_GEMM_BIAS_RELU;
  } else if (params.activation != JKG_ACT_NONE) {
    actual_type = JKG_KERNEL_GEMM_BIAS_ACT;
  }

  Function *func = create_function_stub(module, actual_type, params);
  if (!func) {
    return nullptr;
  }

  LLVMContext &ctx = module->getContext();
  BasicBlock *entry = BasicBlock::Create(ctx, "entry", func);
  builder->SetInsertPoint(entry);

  // Get function arguments
  auto args = func->arg_begin();
  Value *A_ptr = &*args++;
  Value *B_ptr = &*args++;
  Value *C_ptr = &*args++;
  Value *M_arg = &*args++;
  Value *N_arg = &*args++;
  Value *K_arg = &*args++;
  Value *lda = &*args++;
  Value *ldb = &*args++;
  Value *ldc = &*args++;
  Value *alpha = &*args++;

  Value *bias_ptr = nullptr;
  if (params.has_bias) {
    bias_ptr = &*args++;
  }

  Type *int32Ty = Type::getInt32Ty(ctx);
  Type *int64Ty = Type::getInt64Ty(ctx);
  Type *floatTy = Type::getFloatTy(ctx);
  PointerType *floatPtrTy = PointerType::get(floatTy, 0);
  PointerType *voidPtrTy = PointerType::get(Type::getInt8Ty(ctx), 0);

  // Declare KFE function
  FunctionType *kfe_ty = FunctionType::get(
      int32Ty,
      {int32Ty, int32Ty, int32Ty, int64Ty, int64Ty, int64Ty, floatTy,
       floatPtrTy, int64Ty, floatPtrTy, int64Ty, floatPtrTy, int32Ty,
       floatPtrTy, int64Ty, voidPtrTy},
      false);

  Function *kfe_func = module->getFunction("kfe_sgemm_bias_activation");
  if (!kfe_func) {
    kfe_func = Function::Create(kfe_ty, Function::ExternalLinkage,
                                 "kfe_sgemm_bias_activation", module);
  }

  // Map JKG activation to KFE activation
  int kfe_activation = 0; // KFE_ACTIVATION_NONE
  switch (params.activation) {
    case JKG_ACT_RELU: kfe_activation = 1; break; // KFE_ACTIVATION_RELU
    case JKG_ACT_RELU6: kfe_activation = 2; break;
    case JKG_ACT_TANH: kfe_activation = 3; break;
    case JKG_ACT_SIGMOID: kfe_activation = 4; break;
    case JKG_ACT_GELU: kfe_activation = 5; break;
    default: kfe_activation = 0; break;
  }

  Value *layout = ConstantInt::get(int32Ty, 0);
  Value *trans_a = ConstantInt::get(int32Ty, 0);
  Value *trans_b = ConstantInt::get(int32Ty, 0);
  Value *activation_val = ConstantInt::get(int32Ty, kfe_activation);
  Value *stats_null = ConstantPointerNull::get(voidPtrTy);

  if (!bias_ptr) {
    bias_ptr = ConstantPointerNull::get(floatPtrTy);
  }

  Value *M64 = builder->CreateIntCast(M_arg, int64Ty, false);
  Value *N64 = builder->CreateIntCast(N_arg, int64Ty, false);
  Value *K64 = builder->CreateIntCast(K_arg, int64Ty, false);
  Value *lda64 = builder->CreateIntCast(lda, int64Ty, false);
  Value *ldb64 = builder->CreateIntCast(ldb, int64Ty, false);
  Value *ldc64 = builder->CreateIntCast(ldc, int64Ty, false);

  // Call KFE with full fusion
  builder->CreateCall(kfe_func, {layout, trans_a, trans_b, M64, N64, K64, alpha,
                                  A_ptr, lda64, B_ptr, ldb64, bias_ptr, activation_val,
                                  C_ptr, ldc64, stats_null});

  builder->CreateRetVoid();

  // Verify function
  std::string error_str;
  llvm::raw_string_ostream error_stream(error_str);
  if (verifyFunction(*func, &error_stream)) {
    log_error("Function verification failed: %s", error_str.c_str());
    func->eraseFromParent();
    return nullptr;
  }

  return func;
}


/* ========================================================================== */
/* Elementwise Generator Implementation                                       */
/* ========================================================================== */

Function *ElementwiseGenerator::generate(Module *module, LLVMIRBuilder *builder,
                                         const jkg_kernel_params_t &params) {
  LLVMContext &ctx = module->getContext();

  // Create function stub
  Type *floatTy = Type::getFloatTy(ctx);
  llvm::PointerType *floatPtrTy = llvm::PointerType::get(floatTy, 0);
  Type *int64Ty = Type::getInt64Ty(ctx);
  Type *voidTy = Type::getVoidTy(ctx);

  FunctionType *func_type = FunctionType::get(
      voidTy, {floatPtrTy, floatPtrTy, floatPtrTy, int64Ty}, false);

  std::string name = "elementwise_kernel";
  Function *func =
      Function::Create(func_type, Function::ExternalLinkage, name, module);

  BasicBlock *entry = BasicBlock::Create(ctx, "entry", func);
  builder->SetInsertPoint(entry);

  // Extract arguments
  auto args = func->arg_begin();
  Value *A = &*args++;
  A->setName("A");
  Value *B = &*args++;
  B->setName("B");
  Value *C = &*args++;
  C->setName("C");
  Value *N = &*args++;
  N->setName("N");

  // Simple scalar loop
  Value *zero = ConstantInt::get(int64Ty, 0);
  Value *one = ConstantInt::get(int64Ty, 1);

  BasicBlock *loop_header = BasicBlock::Create(ctx, "loop_header", func);
  BasicBlock *loop_body = BasicBlock::Create(ctx, "loop_body", func);
  BasicBlock *loop_end = BasicBlock::Create(ctx, "loop_end", func);

  builder->CreateBr(loop_header);

  builder->SetInsertPoint(loop_header);
  PHINode *i = builder->CreatePHI(int64Ty, 2, "i");
  i->addIncoming(zero, entry);
  Value *cond = builder->CreateICmpULT(i, N);
  builder->CreateCondBr(cond, loop_body, loop_end);

  builder->SetInsertPoint(loop_body);
  Value *a_gep = builder->CreateGEP(floatTy, A, i);
  Value *b_gep = builder->CreateGEP(floatTy, B, i);
  Value *c_gep = builder->CreateGEP(floatTy, C, i);

  Value *a_val = builder->CreateLoad(floatTy, a_gep);
  Value *b_val = builder->CreateLoad(floatTy, b_gep);

  // Perform operation (add or mul based on params)
  Value *result = builder->CreateFAdd(a_val, b_val); // Default to add

  builder->CreateStore(result, c_gep);

  Value *i_next = builder->CreateAdd(i, one);
  i->addIncoming(i_next, loop_body);
  builder->CreateBr(loop_header);

  builder->SetInsertPoint(loop_end);
  builder->CreateRetVoid();

  return func;
}

/* ========================================================================== */
/* Module Optimization (LLVM 20 PassManager)                                  */
/* ========================================================================== */

static void optimize_module(Module *module, int opt_level) {
  // LLVM 20 uses new PassManager
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassBuilder PB;

  // Register analyses
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create optimization pipeline based on level
  llvm::ModulePassManager MPM;
  if (opt_level == 0) {
    MPM = PB.buildO0DefaultPipeline(llvm::OptimizationLevel::O0);
  } else if (opt_level == 1) {
    MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O1);
  } else if (opt_level == 2) {
    MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
  } else {
    MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  }

  // Run optimization passes
  MPM.run(*module, MAM);
}

static void *jitComputeModule(orc::LLJIT *jit, std::unique_ptr<Module> module,
                              const std::string &func_name) {
  if (!jit) {
    log_error("LLJIT instance is null");
    return nullptr;
  }

  // Wrap module in ThreadSafeModule using reference to TSContext
  auto tsm = ThreadSafeModule(std::move(module), *g_jkg_state.ts_context);

  // Add module to JIT
  if (auto err = jit->addIRModule(std::move(tsm))) {
    log_error("Failed to add module to LLJIT: %s",
              toString(std::move(err)).c_str());
    return nullptr;
  }

  // Lookup the function pointer
  auto sym_expected = jit->lookup(func_name);
  if (!sym_expected) {
    log_error("Failed to lookup JIT function: %s", func_name.c_str());
    return nullptr;
  }

  llvm::orc::ExecutorAddr exec_addr = *sym_expected; // Extract ExecutorAddr
  JITTargetAddress addr = exec_addr.getValue();      // Extract raw address
  return reinterpret_cast<void *>(addr);
}
} // namespace jkg_internal

/* ========================================================================== */
/* Kernel Generation API Implementation                                       */
/* ========================================================================== */

using namespace jkg_internal;

int jkg_generate_kernel(jkg_kernel_type_t kernel_type,
                        const jkg_kernel_params_t *params,
                        jkg_kernel_internal_t **out_handle) {

  if (!g_jkg_state.initialized) {
    return JKG_ERR_NOT_INITIALIZED;
  }
  if (!params || !out_handle) {
    return JKG_ERR_INVALID_ARG;
  }
  auto start_time = std::chrono::high_resolution_clock::now();

  // Initialize LLVM if not already done
  if (!g_jkg_state.jit) {
    int ret = initialize_llvm();
    if (ret != JKG_OK) {
      return ret;
    }
  }

  // Determine target ISA
  jkg_isa_t target_isa = g_jkg_state.config.target_isa;
  if (target_isa == JKG_ISA_AUTO) {
    target_isa = select_best_isa(g_jkg_state.available_isa_mask);
  }

  // Check cache
  if (g_jkg_state.config.enable_kernel_cache) {
    KernelCacheKey key;
    key.type = kernel_type;
    key.isa = target_isa;
    key.M = params->M;
    key.N = params->N;
    key.K = params->K;
    key.activation = params->activation;
    key.has_bias = params->has_bias;
    key.has_residual = params->has_residual;

    auto cached = lookup_cached_kernel(key);
    if (cached) {
      *out_handle = reinterpret_cast<jkg_kernel_internal_t *>(cached.get());
      cached->ref_count++;
      return JKG_OK;
    }
  }

  // Create module and builder
  auto ctx_lock = g_jkg_state.ts_context->getLock();
  auto &ctx = *g_jkg_state.ts_context->getContext();
  auto module = std::make_unique<Module>("jit_kernel", ctx);
  auto builder = std::make_unique<LLVMIRBuilder>(ctx);

  // Select appropriate generator
  std::unique_ptr<IRGenerator> generator;
  switch (kernel_type) {
  case JKG_KERNEL_GEMM_TILE:
    generator = std::make_unique<GEMMTileGenerator>();
    break;
  case JKG_KERNEL_GEMM_BIAS:
  case JKG_KERNEL_GEMM_BIAS_RELU:
  case JKG_KERNEL_GEMM_BIAS_ACT:
    generator = std::make_unique<FusedGEMMGenerator>();
    break;
  case JKG_KERNEL_ELEMENTWISE_ADD:
  case JKG_KERNEL_ELEMENTWISE_MUL:
    generator = std::make_unique<ElementwiseGenerator>();
    break;
  default:
    return JKG_ERR_INVALID_ARG;
  }

  // Generate IR
  Function *func = generator->generate(module.get(), builder.get(), *params);
  if (!func) {
    log_error("Failed to generate LLVM IR");
    return JKG_ERR_COMPILATION;
  }

  // Verify IR
  if (llvm::verifyModule(*module, &llvm::errs())) {
    llvm::errs() << "[JKG ERROR] LLVM module verification failed\n";
    module->print(llvm::errs(), nullptr);
    return JKG_ERR_IR_INVALID;
  }

  // Optimize module
  optimize_module(module.get(), g_jkg_state.config.optimization_level);

  // Create kernel handle
  auto handle = std::make_shared<jkg_kernel_impl_t>();
  handle->type = kernel_type;
  handle->params = *params;
  handle->kernel_name = func->getName().str();

  // Compile and get function pointer
  // Note: Actual JIT compilation would happen here
  handle->function_ptr = jitComputeModule(
      g_jkg_state.jit.get(), std::move(module), func->getName().str());
  if (!handle->function_ptr) {
    log_error("JIT compilation failed for kernel: %s",
              func->getName().str().c_str());
    return JKG_ERR_COMPILATION;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();

  g_jkg_state.kernels_generated++;
  g_jkg_state.total_compile_time_us += duration;

  log_info("Kernel generated successfully in %ld us", duration);

  // Cache kernel
  if (g_jkg_state.config.enable_kernel_cache) {
    KernelCacheKey key;
    key.type = kernel_type;
    key.isa = target_isa;
    key.M = params->M;
    key.N = params->N;
    key.K = params->K;
    key.activation = params->activation;
    key.has_bias = params->has_bias;
    key.has_residual = params->has_residual;

    insert_cached_kernel(key, handle);
  }

  *out_handle = reinterpret_cast<jkg_kernel_internal_t *>(handle.get());
  handle->ref_count++;

  return JKG_OK;
}

void *jkg_get_kernel_function(jkg_kernel_internal_t *handle) {
  if (!handle) {
    return nullptr;
  }
  auto impl = handle_to_impl(handle);
  return impl->function_ptr;
}

void jkg_release_kernel(jkg_kernel_internal_t *handle) {
  if (!handle) {
    return;
  }

  auto impl = handle_to_impl(handle);
  impl->ref_count--;
  if (impl->ref_count <= 0) {
    // Kernel will be freed when removed from cache
    log_info("Released kernel: %s", impl->kernel_name.c_str());
  }
}
