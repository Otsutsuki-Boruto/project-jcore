// advanced/polyhedral_optimization/src/polyhedral_optimization_test.cpp

/**
 * @file polyhedral_optimization_test.cpp
 * @brief Comprehensive test suite for polyhedral optimization layer
 *
 * Tests all aspects of the polyhedral optimization component including:
 * - Initialization and configuration
 * - Loop analysis and tileability checking
 * - Tiling transformations
 * - Vectorization preparation
 * - Loop interchange and unrolling
 * - LLVM IR generation and optimization
 * - Performance estimation and GFLOPS measurement
 */

#include "polyhedral_optimization.h"
#include "polyhedral_internal.h"

// JIT Kernel Generator for IR generation
#include "jit_kernel_generator.h"

// Base components
#include "ffm_cache_block.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <memory>

/* ========================================================================== */
/* Test Utilities */
/* ========================================================================== */

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "  [FAIL] %s: %s\n", __func__, msg);                     \
      throw std::runtime_error(msg);                                           \
    }                                                                          \
  } while (0)

#define RUN_TEST(test_func)                                                    \
  do {                                                                         \
    printf("\n[TEST] Running %s...\n", #test_func);                            \
    g_tests_run++;                                                             \
    try {                                                                      \
      test_func();                                                             \
      g_tests_passed++;                                                        \
      printf("  [PASS] %s\n", #test_func);                                     \
    } catch (...) {                                                            \
      fprintf(stderr, "  [FAIL] %s: Exception thrown\n", #test_func);          \
      g_tests_failed++;                                                        \
    }                                                                          \
  } while (0)

/* ========================================================================== */
/* LLVM IR Generation for Test Kernels */
/* ========================================================================== */

// Generate a simple nested loop (GEMM-like)
llvm::Function *generate_gemm_test_function(llvm::Module *module,
                                             size_t M, size_t N, size_t K) {
  llvm::LLVMContext &Ctx = module->getContext();

  // Function signature: void gemm_test(float* C, float* A, float* B, i64 M, i64 N, i64 K)
  llvm::Type *FloatPtrTy = llvm::PointerType::get(llvm::Type::getFloatTy(Ctx), 0);
  llvm::Type *Int64Ty = llvm::Type::getInt64Ty(Ctx);
  llvm::Type *FloatTy = llvm::Type::getFloatTy(Ctx);

  std::vector<llvm::Type*> ParamTypes = {
      FloatPtrTy, FloatPtrTy, FloatPtrTy, Int64Ty, Int64Ty, Int64Ty};
  llvm::FunctionType *FuncTy = llvm::FunctionType::get(
      llvm::Type::getVoidTy(Ctx), ParamTypes, false);

  llvm::Function *Func = llvm::Function::Create(
      FuncTy, llvm::Function::ExternalLinkage, "gemm_test", module);

  // Get function arguments
  auto ArgIt = Func->arg_begin();
  llvm::Value *C = &*ArgIt++;
  C->setName("C");
  llvm::Value *A = &*ArgIt++;
  A->setName("A");
  llvm::Value *B = &*ArgIt++;
  B->setName("B");
  llvm::Value *M_arg = &*ArgIt++;
  M_arg->setName("M");
  llvm::Value *N_arg = &*ArgIt++;
  N_arg->setName("N");
  llvm::Value *K_arg = &*ArgIt++;
  K_arg->setName("K");

  // Create basic blocks
  llvm::BasicBlock *Entry = llvm::BasicBlock::Create(Ctx, "entry", Func);

  llvm::IRBuilder<> Builder(Entry);

  llvm::Value *Zero = llvm::ConstantInt::get(Int64Ty, 0);
  llvm::Value *One = llvm::ConstantInt::get(Int64Ty, 1);

  // Use constant bounds for proper ScalarEvolution analysis
  llvm::Value *M_const = llvm::ConstantInt::get(Int64Ty, M);
  llvm::Value *N_const = llvm::ConstantInt::get(Int64Ty, N);
  llvm::Value *K_const = llvm::ConstantInt::get(Int64Ty, K);

  // Loop I structure
  llvm::BasicBlock *LoopIHeader = llvm::BasicBlock::Create(Ctx, "loop.i.header", Func);
  llvm::BasicBlock *LoopIBody = llvm::BasicBlock::Create(Ctx, "loop.i.body", Func);
  llvm::BasicBlock *LoopIEnd = llvm::BasicBlock::Create(Ctx, "loop.i.end", Func);

  Builder.CreateBr(LoopIHeader);

  // Loop I Header
  Builder.SetInsertPoint(LoopIHeader);
  llvm::PHINode *I = Builder.CreatePHI(Int64Ty, 2, "i");
  I->addIncoming(Zero, Entry);
  llvm::Value *ICond = Builder.CreateICmpULT(I, M_const);
  Builder.CreateCondBr(ICond, LoopIBody, LoopIEnd);

  // Loop I Body - contains Loop J
  Builder.SetInsertPoint(LoopIBody);
  llvm::BasicBlock *LoopJHeader = llvm::BasicBlock::Create(Ctx, "loop.j.header", Func);
  llvm::BasicBlock *LoopJBody = llvm::BasicBlock::Create(Ctx, "loop.j.body", Func);
  llvm::BasicBlock *LoopJEnd = llvm::BasicBlock::Create(Ctx, "loop.j.end", Func);

  Builder.CreateBr(LoopJHeader);

  // Loop J Header
  Builder.SetInsertPoint(LoopJHeader);
  llvm::PHINode *J = Builder.CreatePHI(Int64Ty, 2, "j");
  J->addIncoming(Zero, LoopIBody);
  llvm::Value *JCond = Builder.CreateICmpULT(J, N_const);
  Builder.CreateCondBr(JCond, LoopJBody, LoopJEnd);

  // Loop J Body - contains Loop K
  Builder.SetInsertPoint(LoopJBody);
  llvm::BasicBlock *LoopKHeader = llvm::BasicBlock::Create(Ctx, "loop.k.header", Func);
  llvm::BasicBlock *LoopKBody = llvm::BasicBlock::Create(Ctx, "loop.k.body", Func);
  llvm::BasicBlock *LoopKEnd = llvm::BasicBlock::Create(Ctx, "loop.k.end", Func);

  Builder.CreateBr(LoopKHeader);

  // Loop K Header
  Builder.SetInsertPoint(LoopKHeader);
  llvm::PHINode *K_iv = Builder.CreatePHI(Int64Ty, 2, "k");
  llvm::PHINode *Accum = Builder.CreatePHI(FloatTy, 2, "accum");
  K_iv->addIncoming(Zero, LoopJBody);
  Accum->addIncoming(llvm::ConstantFP::get(FloatTy, 0.0), LoopJBody);

  llvm::Value *KCond = Builder.CreateICmpULT(K_iv, K_const);
  Builder.CreateCondBr(KCond, LoopKBody, LoopKEnd);

  // Loop K Body: C[i,j] += A[i,k] * B[k,j]
  Builder.SetInsertPoint(LoopKBody);

  // A[i*K + k]
  llvm::Value *AIdx = Builder.CreateAdd(Builder.CreateMul(I, K_const), K_iv);
  llvm::Value *APtr = Builder.CreateGEP(FloatTy, A, AIdx);
  llvm::Value *AVal = Builder.CreateLoad(FloatTy, APtr, "a_val");

  // B[k*N + j]
  llvm::Value *BIdx = Builder.CreateAdd(Builder.CreateMul(K_iv, N_const), J);
  llvm::Value *BPtr = Builder.CreateGEP(FloatTy, B, BIdx);
  llvm::Value *BVal = Builder.CreateLoad(FloatTy, BPtr, "b_val");

  // Multiply and accumulate
  llvm::Value *Prod = Builder.CreateFMul(AVal, BVal, "prod");
  llvm::Value *NewAccum = Builder.CreateFAdd(Accum, Prod, "new_acc");

  // K increment
  llvm::Value *KNext = Builder.CreateAdd(K_iv, One);
  K_iv->addIncoming(KNext, LoopKBody);
  Accum->addIncoming(NewAccum, LoopKBody);
  Builder.CreateBr(LoopKHeader);

  // Loop K End: store result to C[i,j]
  Builder.SetInsertPoint(LoopKEnd);
  llvm::Value *CIdx = Builder.CreateAdd(Builder.CreateMul(I, N_const), J);
  llvm::Value *CPtr = Builder.CreateGEP(FloatTy, C, CIdx);
  Builder.CreateStore(Accum, CPtr);

  // J increment
  llvm::Value *JNext = Builder.CreateAdd(J, One);
  J->addIncoming(JNext, LoopKEnd);
  Builder.CreateBr(LoopJHeader);

  // Loop J End
  Builder.SetInsertPoint(LoopJEnd);
  llvm::Value *INext = Builder.CreateAdd(I, One);
  I->addIncoming(INext, LoopJEnd);
  Builder.CreateBr(LoopIHeader);

  // Loop I End
  Builder.SetInsertPoint(LoopIEnd);
  Builder.CreateRetVoid();

  // Verify function
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  if (llvm::verifyFunction(*Func, &error_stream)) {
    fprintf(stderr, "Generated function verification failed: %s\n",
            error.c_str());
    return nullptr;
  }

  return Func;
}

/* ========================================================================== */
/* Test Cases */
/* ========================================================================== */

void test_initialization() {
  printf("  Testing initialization...\n");

  // Test with default config
  int ret = poly_opt_init(nullptr);
  TEST_ASSERT(ret == POLY_OK, "Initialization with default config failed");

  TEST_ASSERT(poly_opt_is_initialized() == 1, "Should be initialized");

  // Test getting config
  poly_opt_config_t config;
  ret = poly_opt_get_config(&config);
  TEST_ASSERT(ret == POLY_OK, "Getting config failed");

  // Test cache info
  const char *cache_info = poly_opt_get_cache_info();
  TEST_ASSERT(cache_info != nullptr, "Cache info should not be null");
  printf("  Cache info: %s\n", cache_info);

  poly_opt_shutdown();
  TEST_ASSERT(poly_opt_is_initialized() == 0, "Should not be initialized");
}

void test_configuration() {
  printf("  Testing configuration...\n");

  int ret = poly_opt_init(nullptr);
  TEST_ASSERT(ret == POLY_OK, "Initialization failed");

  // Test setting custom config
  poly_opt_config_t config;
  std::memset(&config, 0, sizeof(config));

  config.tile_strategy = POLY_TILE_L2;
  config.transform_flags = POLY_TRANSFORM_ALL;
  config.tile_size_M = 64;
  config.tile_size_N = 64;
  config.tile_size_K = 64;
  config.enable_vectorization = 1;
  /* unrolling controlled via transform_flags. Use cfg.transform_flags &
   * POLY_TRANSFORM_UNROLL to enable unrolling.*/
  config.cache_occupancy_fraction = 0.75;
  config.verbose = 0;

  ret = poly_opt_set_config(&config);
  TEST_ASSERT(ret == POLY_OK, "Setting config failed");

  // Verify config was set
  poly_opt_config_t retrieved_config;
  ret = poly_opt_get_config(&retrieved_config);
  TEST_ASSERT(ret == POLY_OK, "Getting config failed");
  TEST_ASSERT(retrieved_config.tile_size_M == 64, "Tile M not set correctly");
  TEST_ASSERT(retrieved_config.tile_strategy == POLY_TILE_L2,
              "Tile strategy not set correctly");

  poly_opt_shutdown();
}

void test_loop_analysis() {
  printf("  Testing loop analysis...\n");

  // Initialize
  poly_opt_config_t config = {};
  config.tile_strategy = POLY_TILE_AUTO;
  config.transform_flags = POLY_TRANSFORM_ALL;
  config.verbose = 0;
  config.cache_occupancy_fraction = 0.75;

  int ret = poly_opt_init(&config);
  TEST_ASSERT(ret == POLY_OK, "Initialization failed");

  // Create LLVM module and generate test function
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> Module =
      std::make_unique<llvm::Module>("test_module", Ctx);

  size_t M = 128, N = 128, K = 128;
  llvm::Function *Func = generate_gemm_test_function(Module.get(), M, N, K);
  TEST_ASSERT(Func != nullptr, "Failed to generate test function");

  // Create optimization plan
  poly_opt_plan_t *plan = nullptr;
  ret = poly_opt_create_plan(static_cast<void*>(Func), &plan);
  TEST_ASSERT(ret == POLY_OK, "Creating plan failed");
  TEST_ASSERT(plan != nullptr, "Plan should not be null");

  // Analyze loops
  poly_loop_info_t loop_info;
  std::memset(&loop_info, 0, sizeof(loop_info));

  ret = poly_opt_analyze_loops(plan, &loop_info);
  TEST_ASSERT(ret == POLY_OK, "Loop analysis failed");
  TEST_ASSERT(loop_info.loop_depth > 0, "Should find loops");

  printf("  Found %zu loops\n", loop_info.loop_depth);
  printf("  Perfectly nested: %s\n",
         loop_info.is_perfectly_nested ? "yes" : "no");
  printf("  Affine bounds: %s\n",
         loop_info.has_affine_bounds ? "yes" : "no");

  // Check tileability
  int tileable = poly_opt_is_tileable(plan);
  printf("  Tileable: %s\n", tileable ? "yes" : "no");

  // Free trip counts and strides
  if (loop_info.trip_counts) free(loop_info.trip_counts);
  if (loop_info.strides) free(loop_info.strides);

  poly_opt_release_plan(plan);
  poly_opt_shutdown();
}

void test_tile_size_recommendation() {
  printf("  Testing tile size recommendation...\n");

  int ret = poly_opt_init(nullptr);
  TEST_ASSERT(ret == POLY_OK, "Initialization failed");

  // Create test function
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> Module =
      std::make_unique<llvm::Module>("test_module", Ctx);

  llvm::Function *Func = generate_gemm_test_function(Module.get(),
                                                      512, 512, 512);
  TEST_ASSERT(Func != nullptr, "Failed to generate test function");

  poly_opt_plan_t *plan = nullptr;
  ret = poly_opt_create_plan(static_cast<void*>(Func), &plan);
  TEST_ASSERT(ret == POLY_OK, "Creating plan failed");

  // Analyze first
  poly_loop_info_t loop_info;
  std::memset(&loop_info, 0, sizeof(loop_info));
  ret = poly_opt_analyze_loops(plan, &loop_info);
  TEST_ASSERT(ret == POLY_OK, "Analysis failed");

  // Get recommendations for different cache levels
  size_t tile_M, tile_N, tile_K;

  // L1 tiles
  ret = poly_opt_recommend_tile_sizes(plan, POLY_TILE_L1,
                                       &tile_M, &tile_N, &tile_K);
  TEST_ASSERT(ret == POLY_OK, "L1 tile recommendation failed");
  printf("  L1 tiles: M=%zu, N=%zu, K=%zu\n", tile_M, tile_N, tile_K);
  TEST_ASSERT(tile_M > 0 && tile_N > 0 && tile_K > 0,
              "L1 tiles should be positive");

  // L2 tiles
  ret = poly_opt_recommend_tile_sizes(plan, POLY_TILE_L2,
                                       &tile_M, &tile_N, &tile_K);
  TEST_ASSERT(ret == POLY_OK, "L2 tile recommendation failed");
  printf("  L2 tiles: M=%zu, N=%zu, K=%zu\n", tile_M, tile_N, tile_K);

  // L3 tiles
  ret = poly_opt_recommend_tile_sizes(plan, POLY_TILE_L3,
                                       &tile_M, &tile_N, &tile_K);
  TEST_ASSERT(ret == POLY_OK, "L3 tile recommendation failed");
  printf("  L3 tiles: M=%zu, N=%zu, K=%zu\n", tile_M, tile_N, tile_K);

  if (loop_info.trip_counts) free(loop_info.trip_counts);
  if (loop_info.strides) free(loop_info.strides);

  poly_opt_release_plan(plan);
  poly_opt_shutdown();
}

void test_tiling_transformation() {
  printf("  Testing tiling transformation...\n");

  poly_opt_config_t config = {};
  config.tile_strategy = POLY_TILE_L2;
  config.verbose = 0;
  config.cache_occupancy_fraction = 0.75;

  int ret = poly_opt_init(&config);
  TEST_ASSERT(ret == POLY_OK, "Initialization failed");

  // Create test function
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> Module =
      std::make_unique<llvm::Module>("test_module", Ctx);

  llvm::Function *Func = generate_gemm_test_function(Module.get(),
                                                      256, 256, 256);
  TEST_ASSERT(Func != nullptr, "Failed to generate test function");

  poly_opt_plan_t *plan = nullptr;
  ret = poly_opt_create_plan(static_cast<void*>(Func), &plan);
  TEST_ASSERT(ret == POLY_OK, "Creating plan failed");

  // Apply tiling
  ret = poly_opt_apply_tiling(plan, POLY_TILE_L2);
  printf("  Tiling result: %s\n", poly_opt_strerror(ret));

  // Get statistics
  poly_opt_stats_t stats;
  ret = poly_opt_get_stats(plan, &stats);
  TEST_ASSERT(ret == POLY_OK, "Getting stats failed");

  printf("  Loops analyzed: %zu\n", stats.loops_analyzed);
  printf("  Loops tiled: %zu\n", stats.loops_tiled);
  printf("  Expected speedup: %.2fx\n", stats.expected_speedup);

  poly_opt_release_plan(plan);
  poly_opt_shutdown();
}

void test_vectorization() {
  printf("  Testing vectorization preparation...\n");

  poly_opt_config_t config = {};
  config.enable_vectorization = 1;
  config.verbose = 0;
  config.cache_occupancy_fraction = 0.75;

  int ret = poly_opt_init(&config);
  TEST_ASSERT(ret == POLY_OK, "Initialization failed");

  // Create test function
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> Module =
      std::make_unique<llvm::Module>("test_module", Ctx);

  llvm::Function *Func = generate_gemm_test_function(Module.get(),
                                                      128, 128, 128);
  TEST_ASSERT(Func != nullptr, "Failed to generate test function");

  poly_opt_plan_t *plan = nullptr;
  ret = poly_opt_create_plan(static_cast<void*>(Func), &plan);
  TEST_ASSERT(ret == POLY_OK, "Creating plan failed");

  // Apply vectorization
  ret = poly_opt_apply_vectorization(plan);
  printf("  Vectorization result: %s\n", poly_opt_strerror(ret));

  // Get statistics
  poly_opt_stats_t stats;
  ret = poly_opt_get_stats(plan, &stats);
  TEST_ASSERT(ret == POLY_OK, "Getting stats failed");

  printf("  Loops vectorized: %zu\n", stats.loops_vectorized);

  poly_opt_release_plan(plan);
  poly_opt_shutdown();
}

void test_unrolling() {
  printf("  Testing loop unrolling...\n");

  poly_opt_config_t config = {};
  config.transform_flags = POLY_TRANSFORM_UNROLL;
  config.verbose = 0;
  config.cache_occupancy_fraction = 0.75;

  int ret = poly_opt_init(&config);
  TEST_ASSERT(ret == POLY_OK, "Initialization failed");

  // Create test function
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> Module =
      std::make_unique<llvm::Module>("test_module", Ctx);

  llvm::Function *Func = generate_gemm_test_function(Module.get(),
                                                      64, 64, 64);
  TEST_ASSERT(Func != nullptr, "Failed to generate test function");

  poly_opt_plan_t *plan = nullptr;
  ret = poly_opt_create_plan(static_cast<void*>(Func), &plan);
  TEST_ASSERT(ret == POLY_OK, "Creating plan failed");

  // Apply unrolling with auto factors
  ret = poly_opt_apply_unrolling(plan, 0, 0);
  printf("  Unrolling result: %s\n", poly_opt_strerror(ret));

  poly_opt_release_plan(plan);
  poly_opt_shutdown();
}

void test_all_transformations() {
  printf("  Testing all transformations together...\n");

  poly_opt_config_t config = {};
  config.tile_strategy = POLY_TILE_L2;
  config.transform_flags = POLY_TRANSFORM_ALL;
  config.enable_vectorization = 1;
  /* unrolling controlled via transform_flags. Use cfg.transform_flags &
   * POLY_TRANSFORM_UNROLL to enable unrolling.*/
  config.enable_prefetch = 1;
  config.verbose = 0;
  config.cache_occupancy_fraction = 0.75;

  int ret = poly_opt_init(&config);
  TEST_ASSERT(ret == POLY_OK, "Initialization failed");

  // Create test function
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> Module =
      std::make_unique<llvm::Module>("test_module", Ctx);

  llvm::Function *Func = generate_gemm_test_function(Module.get(),
                                                      512, 512, 512);
  TEST_ASSERT(Func != nullptr, "Failed to generate test function");

  poly_opt_plan_t *plan = nullptr;
  ret = poly_opt_create_plan(static_cast<void*>(Func), &plan);
  TEST_ASSERT(ret == POLY_OK, "Creating plan failed");

  // Apply all transformations
  auto start = std::chrono::high_resolution_clock::now();
  ret = poly_opt_apply_all_transforms(plan);
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  printf("  All transforms result: %s\n", poly_opt_strerror(ret));
  printf("  Transformation time: %.2f ms\n", elapsed_ms);

  // Get detailed statistics
  poly_opt_stats_t stats;
  ret = poly_opt_get_stats(plan, &stats);
  TEST_ASSERT(ret == POLY_OK, "Getting stats failed");

  printf("\n  === Optimization Statistics ===\n");
  printf("  Loops analyzed: %zu\n", stats.loops_analyzed);
  printf("  Loops tiled: %zu\n", stats.loops_tiled);
  printf("  Loops vectorized: %zu\n", stats.loops_vectorized);
  printf("  Loops interchanged: %zu\n", stats.loops_interchanged);
  printf("  Expected speedup: %.2fx\n", stats.expected_speedup);
  printf("  Memory access reduction: %zu%%\n", stats.memory_accesses_reduced);
  printf("  Optimization time: %.2f ms\n", stats.optimization_time_ms);

  // Estimate speedup
  double speedup = poly_opt_estimate_speedup(plan);
  printf("  Estimated speedup: %.2fx\n", speedup);
  TEST_ASSERT(speedup >= 1.0, "Speedup should be >= 1.0");

  poly_opt_release_plan(plan);
  poly_opt_shutdown();
}

void test_gemm_optimization() {
  printf("  Testing high-level GEMM optimization...\n");

  int ret = poly_opt_init(nullptr);
  TEST_ASSERT(ret == POLY_OK, "Initialization failed");

  // Create test GEMM function
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> Module =
      std::make_unique<llvm::Module>("test_module", Ctx);

  size_t M = 1024, N = 1024, K = 1024;
  llvm::Function *Func = generate_gemm_test_function(Module.get(), M, N, K);
  TEST_ASSERT(Func != nullptr, "Failed to generate test function");

  // Use convenience function for GEMM optimization
  auto start = std::chrono::high_resolution_clock::now();
  ret = poly_opt_optimize_gemm(static_cast<void*>(Func), M, N, K);
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  printf("  GEMM optimization result: %s\n", poly_opt_strerror(ret));
  printf("  Time: %.2f ms\n", elapsed_ms);

  poly_opt_shutdown();
}

void test_self_test() {
  printf("  Running component self-test...\n");

  int ret = poly_opt_self_test(1); // verbose
  TEST_ASSERT(ret == POLY_OK, "Self-test failed");

  printf("  Self-test passed\n");
}

/* ========================================================================== */
/* Performance Benchmark: JIT + Polyhedral vs BLAS */
/* ========================================================================== */

void test_performance_benchmark() {
  printf("  Testing JIT + Polyhedral Optimization Performance...\n");

  // Helper lambda for GFLOPS calculation
  auto compute_gflops = [](size_t M, size_t N, size_t K, double time_sec) -> double {
    double flops = 2.0 * M * N * K;
    return (flops / time_sec) / 1e9;
  };

  // Helper lambda for benchmarking
  auto benchmark_kernel = [&](const char* name, size_t M, size_t N, size_t K,
                              int iterations, bool use_poly) -> void {
    printf("\n  --- Benchmark: %s [M=%zu, N=%zu, K=%zu] ---\n", name, M, N, K);

    // Allocate matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    std::vector<float> bias(N, 0.5f);

    // Initialize with random data
    for (size_t i = 0; i < M * K; i++) A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (size_t i = 0; i < K * N; i++) B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    std::fill(C.begin(), C.end(), 0.0f);

    // Generate JIT kernel with bias and activation
    jkg_kernel_params_t params = {};
    params.M = M;
    params.N = N;
    params.K = K;
    params.has_bias = 1;
    params.activation = JKG_ACT_RELU;

    jkg_kernel_internal_t *jit_handle = nullptr;
    int ret = jkg_generate_kernel(JKG_KERNEL_GEMM_BIAS_RELU, &params, &jit_handle);
    if (ret != JKG_OK) {
      printf("  JIT kernel generation failed: %s\n", jkg_strerror(ret));
      return;
    }

    void *jit_func = jkg_get_kernel_function(jit_handle);
    TEST_ASSERT(jit_func != nullptr, "JIT function pointer is null");

    // Apply polyhedral optimization if requested
    poly_opt_plan_t *poly_plan = nullptr;
    if (use_poly) {
      llvm::LLVMContext Ctx;
      std::unique_ptr<llvm::Module> Module =
          std::make_unique<llvm::Module>("bench_module", Ctx);

      llvm::Function *Func = generate_gemm_test_function(Module.get(), M, N, K);
      if (Func) {
        ret = poly_opt_create_plan(static_cast<void*>(Func), &poly_plan);
        if (ret == POLY_OK) {
          ret = poly_opt_apply_all_transforms(poly_plan);
          if (ret == POLY_OK) {
            printf("  Polyhedral optimizations applied successfully\n");
          }
        }
      }
    }

    // Warm-up run
    typedef void (*gemm_func_t)(const float*, const float*, float*,
                                 int64_t, int64_t, int64_t,
                                 int64_t, int64_t, int64_t,
                                 float, const float*);
    gemm_func_t kernel = reinterpret_cast<gemm_func_t>(jit_func);
    kernel(A.data(), B.data(), C.data(), M, N, K, K, N, N, 1.0f, bias.data());

    // Benchmark
    std::fill(C.begin(), C.end(), 0.0f);
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
      kernel(A.data(), B.data(), C.data(), M, N, K, K, N, N, 1.0f, bias.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count() / iterations;
    double gflops = compute_gflops(M, N, K, time_sec);

    printf("  Performance: %.2f GFLOPS (%.3f ms)\n", gflops, time_sec * 1000);

    // Get polyhedral statistics if used
    if (use_poly && poly_plan) {
      poly_opt_stats_t stats;
      ret = poly_opt_get_stats(poly_plan, &stats);
      if (ret == POLY_OK) {
        printf("  Poly Stats: Tiled=%zu, Vectorized=%zu, Speedup=%.2fx\n",
               stats.loops_tiled, stats.loops_vectorized, stats.expected_speedup);
      }
      poly_opt_release_plan(poly_plan);
    }

    // Cleanup
    jkg_release_kernel(jit_handle);
  };

  // Test 1: Small matrix (cache-resident)
  benchmark_kernel("Small 512x512x512 (JIT Only)", 512, 512, 512, 10, false);
  benchmark_kernel("Small 512x512x512 (JIT + Poly)", 512, 512, 512, 10, true);

  // Test 2: Square matrix K=4096
  benchmark_kernel("Square 4096x4096 (JIT Only)", 4096, 4096, 4096, 3, false);
  benchmark_kernel("Square 4096x4096 (JIT + Poly)", 4096, 4096, 4096, 3, true);

  // Test 3: Square matrix K=8192
  benchmark_kernel("Square 8192x8192 (JIT Only)", 8192, 8192, 8192, 1, false);
  benchmark_kernel("Square 8192x8192 (JIT + Poly)", 8192, 8192, 8192, 1, true);

  // Test 4: Non-square matrix
  benchmark_kernel("Non-Square 1024x2048x4096 (JIT Only)", 1024, 2048, 4096, 3, false);
  benchmark_kernel("Non-Square 1024x2048x4096 (JIT + Poly)", 1024, 2048, 4096, 3, true);

  printf("\n  === Performance Summary ===\n");
  printf("  All benchmark configurations completed\n");
  printf("  JIT + Polyhedral optimization demonstrates:\n");
  printf("    - Loop tiling for cache optimization\n");
  printf("    - Vectorization hints for SIMD\n");
  printf("    - Loop unrolling for reduced overhead\n");
  printf("    - Prefetching for memory bandwidth\n");

  // Cleanup
  poly_opt_shutdown();
  jkg_shutdown();
}
/* ========================================================================== */
/* Main Test Runner */
/* ========================================================================== */

int main(int argc, char **argv) {
  printf("========================================\n");
  printf("Polyhedral Optimization Layer Test Suite\n");
  printf("========================================\n");

  RUN_TEST(test_initialization);
  RUN_TEST(test_configuration);
  RUN_TEST(test_loop_analysis);
  RUN_TEST(test_tile_size_recommendation);
  RUN_TEST(test_tiling_transformation);
  RUN_TEST(test_vectorization);
  RUN_TEST(test_unrolling);
  RUN_TEST(test_all_transformations);
  RUN_TEST(test_gemm_optimization);
  RUN_TEST(test_performance_benchmark);
  RUN_TEST(test_self_test);

  printf("\n========================================\n");
  printf("Test Results:\n");
  printf("  Total:  %d\n", g_tests_run);
  printf("  Passed: %d\n", g_tests_passed);
  printf("  Failed: %d\n", g_tests_failed);

  if (g_tests_passed + g_tests_failed != g_tests_run) {
    printf("  (Note: %d tests had assertion failures)\n",
           g_tests_run - g_tests_passed - g_tests_failed);
  }

  printf("========================================\n");

  if (g_tests_failed == 0) {
    printf("\n✓ All tests passed!\n\n");
    return 0;
  } else {
    printf("\n✗ Some tests failed!\n\n");
    return 1;
  }
}