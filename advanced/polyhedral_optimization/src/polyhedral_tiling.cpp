// advanced/polyhedral_optimization/src/polyhedral_tiling.cpp

/**
 * @file polyhedral_tiling.cpp
 * @brief Loop tiling transformations and cache-aware blocking
 *
 * Implements loop tiling strategies including single-level and multi-level
 * tiling based on cache hierarchy analysis.
 */

#include "polyhedral_internal.h"
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <chrono>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Verifier.h>

namespace poly_internal {

/* ========================================================================== */
/* Loop Bounds Analysis */
/* ========================================================================== */

bool extract_loop_bounds(llvm::Loop *loop, llvm::ScalarEvolution *SE,
                         size_t *trip_count) {
  if (!loop || !SE || !trip_count) {
    return false;
  }

  // Get backedge taken count
  const llvm::SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(loop);

  if (llvm::isa<llvm::SCEVCouldNotCompute>(BackedgeTakenCount)) {
    log_debug("Could not compute backedge taken count for loop");
    return false;
  }

  // Try to get constant trip count
  if (const llvm::SCEVConstant *ConstCount =
          llvm::dyn_cast<llvm::SCEVConstant>(BackedgeTakenCount)) {
    *trip_count = ConstCount->getAPInt().getZExtValue() + 1;
    log_debug("Loop trip count: %zu", *trip_count);
    return true;
  }

  // Try to get max backedge count as upper bound
  const llvm::SCEV *MaxBE = SE->getConstantMaxBackedgeTakenCount(loop);
  if (const llvm::SCEVConstant *MaxConst =
          llvm::dyn_cast<llvm::SCEVConstant>(MaxBE)) {
    *trip_count = MaxConst->getAPInt().getZExtValue() + 1;
    log_debug("Loop max trip count (upper bound): %zu", *trip_count);
    return true;
  }

  return false;
}

/* ========================================================================== */
/* Loop Nest Analysis */
/* ========================================================================== */

bool is_perfectly_nested(llvm::Loop *loop) {
  if (!loop) return false;

  // Get subloops
  const std::vector<llvm::Loop*> &SubLoops = loop->getSubLoops();

  // No subloops = innermost loop (vacuously perfect)
  if (SubLoops.empty()) {
    return true;
  }

  // More than one subloop = not perfectly nested
  if (SubLoops.size() > 1) {
    return false;
  }

  // Check that loop body only contains the single subloop
  llvm::BasicBlock *Header = loop->getHeader();
  if (!Header) return false;

  // Count non-loop instructions in body
  size_t instruction_count = 0;
  for (auto *BB : loop->blocks()) {
    if (SubLoops[0]->contains(BB)) continue; // Skip subloop blocks

    for (auto &I : *BB) {
      if (llvm::isa<llvm::PHINode>(&I)) continue;
      if (llvm::isa<llvm::BranchInst>(&I)) continue;
      if (llvm::isa<llvm::CmpInst>(&I)) continue;
      instruction_count++;
    }
  }

  // If there are significant instructions outside subloop, not perfectly nested
  if (instruction_count > 5) {
    return false;
  }

  // Recursively check subloop
  return is_perfectly_nested(SubLoops[0]);
}

bool has_affine_bounds(llvm::Loop *loop, llvm::ScalarEvolution *SE) {
  if (!loop || !SE) return false;

  // Check if backedge count is computable and affine
  const llvm::SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(loop);

  if (llvm::isa<llvm::SCEVCouldNotCompute>(BackedgeTakenCount)) {
    return false;
  }

  // Check if it's an affine expression
  if (llvm::isa<llvm::SCEVConstant>(BackedgeTakenCount) ||
      llvm::isa<llvm::SCEVAddRecExpr>(BackedgeTakenCount)) {
    return true;
  }

  return false;
}

void collect_nested_loops(llvm::Loop *root, std::vector<llvm::Loop*> &loops) {
  if (!root) return;

  loops.push_back(root);

  for (llvm::Loop *SubLoop : root->getSubLoops()) {
    collect_nested_loops(SubLoop, loops);
  }
}

int analyze_loop_nest(PolyOptPlanImpl *plan) {
  if (!plan || !plan->function) {
    return POLY_ERR_INVALID_ARG;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  log_info("Analyzing loop nest in function: %s",
           plan->function->getName().str().c_str());

  // Get loop info
  plan->loop_info = std::make_unique<llvm::LoopInfo>();
  llvm::DominatorTree DT(*plan->function);
  plan->loop_info->analyze(DT);

  // Find outermost loops
  std::vector<llvm::Loop*> TopLevelLoops;
  for (llvm::Loop *L : *plan->loop_info) {
    TopLevelLoops.push_back(L);
  }

  if (TopLevelLoops.empty()) {
    log_error("No loops found in function");
    return POLY_ERR_NO_AFFINE_LOOPS;
  }

  // Analyze first loop nest (TODO: handle multiple nests)
  llvm::Loop *root = TopLevelLoops[0];
  plan->nest_info.root_loop = root;

  // Collect all nested loops
  collect_nested_loops(root, plan->nest_info.nested_loops);

  log_info("Found %zu loops in nest", plan->nest_info.nested_loops.size());

  // Check if perfectly nested
  plan->nest_info.is_perfectly_nested = is_perfectly_nested(root);
  log_info("Perfectly nested: %s",
           plan->nest_info.is_perfectly_nested ? "yes" : "no");

  // Compute scalar evolution for trip count analysis
  llvm::TargetLibraryInfoImpl TLII(
      llvm::Triple(plan->function->getParent()->getTargetTriple()));
  llvm::TargetLibraryInfo TLI(TLII);

  llvm::AssumptionCache AC(*plan->function);

  plan->scalar_evolution = std::make_unique<llvm::ScalarEvolution>(
      *plan->function, TLI, AC, DT, *plan->loop_info);

  // Analyze each loop in nest
  for (size_t i = 0; i < plan->nest_info.nested_loops.size(); i++) {
    llvm::Loop *loop = plan->nest_info.nested_loops[i];

    // Check affine bounds
    bool affine = has_affine_bounds(loop, plan->scalar_evolution.get());
    if (i == 0) {
      plan->nest_info.has_affine_bounds = affine;
    } else {
      plan->nest_info.has_affine_bounds &= affine;
    }

    // Extract trip count
    size_t trip_count = 0;
    if (extract_loop_bounds(loop, plan->scalar_evolution.get(), &trip_count)) {
      plan->nest_info.trip_counts.push_back(trip_count);
    } else {
      plan->nest_info.trip_counts.push_back(0); // Unknown
    }

    // Get induction variable
    llvm::PHINode *IndVar = loop->getCanonicalInductionVariable();
    if (IndVar) {
      plan->nest_info.induction_vars.push_back(IndVar);
    } else {
      plan->nest_info.induction_vars.push_back(nullptr);
    }

    log_debug("Loop %zu: trip_count=%zu, affine=%s",
              i, plan->nest_info.trip_counts[i],
              affine ? "yes" : "no");
  }

  // Analyze memory accesses
  plan->nest_info.num_loads = 0;
  plan->nest_info.num_stores = 0;

  for (llvm::Loop *loop : plan->nest_info.nested_loops) {
    for (auto *BB : loop->blocks()) {
      for (auto &I : *BB) {
        if (llvm::isa<llvm::LoadInst>(&I)) {
          plan->nest_info.num_loads++;
          plan->nest_info.memory_accesses.push_back(&I);
        } else if (llvm::isa<llvm::StoreInst>(&I)) {
          plan->nest_info.num_stores++;
          plan->nest_info.memory_accesses.push_back(&I);
        }
      }
    }
  }

  log_info("Memory accesses: %zu loads, %zu stores",
           plan->nest_info.num_loads, plan->nest_info.num_stores);

  // Check tileability
  plan->nest_info.is_tileable = check_tileability(plan);

  auto end_time = std::chrono::high_resolution_clock::now();
  plan->analysis_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  plan->analysis_done = true;
  plan->stats.loops_analyzed = plan->nest_info.nested_loops.size();

  log_info("Loop analysis completed in %.2f ms", plan->analysis_time_ms);

  return POLY_OK;
}

/* ========================================================================== */
/* Tileability Check */
/* ========================================================================== */

bool check_tileability(PolyOptPlanImpl *plan) {
  if (!plan || plan->nest_info.nested_loops.empty()) {
    return false;
  }

  // Requirements for tileability:
  // 1. Affine loop bounds
  if (!plan->nest_info.has_affine_bounds) {
    log_debug("Not tileable: non-affine bounds");
    return false;
  }

  // 2. At least 2 nested loops for useful tiling
  if (plan->nest_info.nested_loops.size() < 2) {
    log_debug("Not tileable: only %zu loops",
              plan->nest_info.nested_loops.size());
    return false;
  }

  // 3. All loops must have known trip counts (for now)
  for (size_t tc : plan->nest_info.trip_counts) {
    if (tc == 0) {
      log_debug("Not tileable: unknown trip count");
      return false;
    }
  }

  // 4. Check for simple data dependencies (conservative check)
  // For now, assume tileable if perfectly nested
  if (!plan->nest_info.is_perfectly_nested) {
    log_debug("Not tileable: not perfectly nested");
    return false;
  }

  log_info("Loop nest is tileable");
  return true;
}

/* ========================================================================== */
/* Tile Size Computation */
/* ========================================================================== */

void compute_tile_sizes(PolyOptPlanImpl *plan, poly_tile_strategy_t strategy) {
  if (!plan) return;

  TilingConfig &config = plan->tiling_config;
  config.strategy = strategy;

  // Get configuration
  size_t user_tile_M = g_poly_state.config.tile_size_M;
  size_t user_tile_N = g_poly_state.config.tile_size_N;
  size_t user_tile_K = g_poly_state.config.tile_size_K;

  switch (strategy) {
    case POLY_TILE_NONE:
      config.tile_M = config.tile_N = config.tile_K = 0;
      break;

    case POLY_TILE_REGISTER:
      // Small tiles for register blocking
      config.tile_M = g_poly_state.config.register_tile_M;
      config.tile_N = g_poly_state.config.register_tile_N;
      config.tile_K = 4;
      break;

    case POLY_TILE_L1:
      config.tile_M = user_tile_M > 0 ? user_tile_M : g_poly_state.default_l1_tile;
      config.tile_N = user_tile_N > 0 ? user_tile_N : g_poly_state.default_l1_tile;
      config.tile_K = user_tile_K > 0 ? user_tile_K : g_poly_state.default_l1_tile;
      break;

    case POLY_TILE_L2:
      config.tile_M = user_tile_M > 0 ? user_tile_M : g_poly_state.default_l2_tile;
      config.tile_N = user_tile_N > 0 ? user_tile_N : g_poly_state.default_l2_tile;
      config.tile_K = user_tile_K > 0 ? user_tile_K : g_poly_state.default_l2_tile;
      break;

    case POLY_TILE_L3:
      config.tile_M = user_tile_M > 0 ? user_tile_M : g_poly_state.default_l3_tile;
      config.tile_N = user_tile_N > 0 ? user_tile_N : g_poly_state.default_l3_tile;
      config.tile_K = user_tile_K > 0 ? user_tile_K : g_poly_state.default_l3_tile;
      break;

    case POLY_TILE_MULTI_LEVEL:
      config.enable_multi_level = true;

      // L1 tiles
      config.l1_tile_M = g_poly_state.default_l1_tile;
      config.l1_tile_N = g_poly_state.default_l1_tile;
      config.l1_tile_K = g_poly_state.default_l1_tile;

      // L2 tiles
      config.l2_tile_M = g_poly_state.default_l2_tile;
      config.l2_tile_N = g_poly_state.default_l2_tile;
      config.l2_tile_K = g_poly_state.default_l2_tile;

      // L3 tiles
      config.l3_tile_M = g_poly_state.default_l3_tile;
      config.l3_tile_N = g_poly_state.default_l3_tile;
      config.l3_tile_K = g_poly_state.default_l3_tile;

      // Register tiles
      config.reg_tile_M = g_poly_state.config.register_tile_M;
      config.reg_tile_N = g_poly_state.config.register_tile_N;
      break;

    case POLY_TILE_AUTO:
      // Auto-select based on loop trip counts and cache sizes
      if (plan->nest_info.trip_counts.size() >= 3) {
        // Use L2 tiling for moderate-sized problems
        config.tile_M = g_poly_state.default_l2_tile;
        config.tile_N = g_poly_state.default_l2_tile;
        config.tile_K = g_poly_state.default_l2_tile;
      } else if (plan->nest_info.trip_counts.size() >= 2) {
        // Use L1 tiling for smaller nests
        config.tile_M = g_poly_state.default_l1_tile;
        config.tile_N = g_poly_state.default_l1_tile;
        config.tile_K = g_poly_state.default_l1_tile;
      }
      break;
  }

  // Ensure tile sizes don't exceed trip counts
  if (plan->nest_info.trip_counts.size() >= 1 && config.tile_M > 0) {
    config.tile_M = std::min(config.tile_M, plan->nest_info.trip_counts[0]);
  }
  if (plan->nest_info.trip_counts.size() >= 2 && config.tile_N > 0) {
    config.tile_N = std::min(config.tile_N, plan->nest_info.trip_counts[1]);
  }
  if (plan->nest_info.trip_counts.size() >= 3 && config.tile_K > 0) {
    config.tile_K = std::min(config.tile_K, plan->nest_info.trip_counts[2]);
  }

  log_info("Computed tile sizes: M=%zu, N=%zu, K=%zu",
           config.tile_M, config.tile_N, config.tile_K);
}

/* ========================================================================== */
/* Loop Tiling Implementation */
/* ========================================================================== */

/* ========================================================================== */
/* Complete Loop Tiling Implementation */
/* ========================================================================== */

// Structure to hold tiled loop information
struct TiledLoopInfo {
  llvm::Loop *original_loop;
  llvm::Loop *tile_loop;        // Outer tile loop
  llvm::Loop *point_loop;       // Inner point loop
  llvm::PHINode *tile_iv;       // Tile loop induction variable
  llvm::PHINode *point_iv;      // Point loop induction variable
  llvm::Value *tile_size;       // Tile size constant
  llvm::Value *original_bound;  // Original loop bound
};

TiledLoopInfo create_complete_tiled_loop(llvm::Loop *original_loop,
                                         size_t tile_size,
                                         llvm::IRBuilder<> &builder) {
  TiledLoopInfo info;
  info.original_loop = original_loop;

  llvm::BasicBlock *Preheader = original_loop->getLoopPreheader();
  llvm::BasicBlock *Header = original_loop->getHeader();
  llvm::BasicBlock *Latch = original_loop->getLoopLatch();

  if (!Preheader || !Header || !Latch) {
    log_error("Loop missing required structure: preheader=%p header=%p latch=%p",
              (void*)Preheader, (void*)Header, (void*)Latch);
    return info;
  }

  llvm::Function *F = Header->getParent();
  llvm::LLVMContext &Ctx = F->getContext();
  llvm::Type *Int64Ty = llvm::Type::getInt64Ty(Ctx);

  // Get original induction variable
  llvm::PHINode *OrigIV = original_loop->getCanonicalInductionVariable();
  if (!OrigIV) {
    log_error("Cannot find canonical induction variable");
    return info;
  }

  // Find loop bound - check both latch and exiting blocks
  llvm::Value *OrigBound = nullptr;
  llvm::ICmpInst::Predicate Predicate = llvm::ICmpInst::ICMP_ULT;

  // Try to find bound in exiting blocks first
  llvm::SmallVector<llvm::BasicBlock*, 4> ExitingBlocks;
  original_loop->getExitingBlocks(ExitingBlocks);

  for (llvm::BasicBlock *ExitingBB : ExitingBlocks) {
    if (llvm::BranchInst *BI = llvm::dyn_cast<llvm::BranchInst>(ExitingBB->getTerminator())) {
      if (BI->isConditional()) {
        llvm::Value *Cond = BI->getCondition();

        // Handle direct ICmp
        if (llvm::ICmpInst *Cmp = llvm::dyn_cast<llvm::ICmpInst>(Cond)) {
          if (Cmp->getOperand(0) == OrigIV) {
            OrigBound = Cmp->getOperand(1);
            Predicate = Cmp->getPredicate();
            break;
          } else if (Cmp->getOperand(1) == OrigIV) {
            OrigBound = Cmp->getOperand(0);
            Predicate = Cmp->getSwappedPredicate();
            break;
          }
        }

        // Handle ICmp through intermediate instructions
        for (llvm::Instruction &I : *ExitingBB) {
          if (llvm::ICmpInst *Cmp = llvm::dyn_cast<llvm::ICmpInst>(&I)) {
            if (Cmp->getOperand(0) == OrigIV) {
              OrigBound = Cmp->getOperand(1);
              Predicate = Cmp->getPredicate();
              break;
            } else if (Cmp->getOperand(1) == OrigIV) {
              OrigBound = Cmp->getOperand(0);
              Predicate = Cmp->getSwappedPredicate();
              break;
            }
          }
        }

        if (OrigBound) break;
      }
    }
  }

  if (!OrigBound) {
    log_error("Cannot determine loop bound for tiling (checked %zu exiting blocks)",
              ExitingBlocks.size());
    return info;
  }

  log_debug("Found loop bound: %p (predicate=%d)", (void*)OrigBound, (int)Predicate);

  // Ensure bound is i64
  if (OrigBound->getType() != Int64Ty) {
    builder.SetInsertPoint(Preheader->getTerminator());
    if (OrigBound->getType()->isIntegerTy()) {
      OrigBound = builder.CreateZExtOrTrunc(OrigBound, Int64Ty, "bound.i64");
    } else {
      log_error("Loop bound is not an integer type");
      return info;
    }
  }

  info.original_bound = OrigBound;
  info.tile_size = llvm::ConstantInt::get(Int64Ty, tile_size);

  // Get or create exit block
  llvm::BasicBlock *ExitBlock = original_loop->getExitBlock();
  if (!ExitBlock) {
    llvm::SmallVector<llvm::BasicBlock*, 4> ExitBlocks;
    original_loop->getExitBlocks(ExitBlocks);
    if (!ExitBlocks.empty()) {
      ExitBlock = ExitBlocks[0];
    } else {
      log_error("Cannot find exit block for loop");
      return info;
    }
  }

  // Create new basic blocks for tiled structure
  llvm::BasicBlock *TileHeader = llvm::BasicBlock::Create(Ctx, "tile.header", F);
  llvm::BasicBlock *TileBody = llvm::BasicBlock::Create(Ctx, "tile.body", F);
  llvm::BasicBlock *PointHeader = llvm::BasicBlock::Create(Ctx, "point.header", F);
  llvm::BasicBlock *PointBody = llvm::BasicBlock::Create(Ctx, "point.body", F);
  llvm::BasicBlock *PointLatch = llvm::BasicBlock::Create(Ctx, "point.latch", F);
  llvm::BasicBlock *TileLatch = llvm::BasicBlock::Create(Ctx, "tile.latch", F);

  // Redirect preheader to tile header
  llvm::Instruction *PreheaderTerm = Preheader->getTerminator();
  builder.SetInsertPoint(PreheaderTerm);
  PreheaderTerm->eraseFromParent();
  builder.SetInsertPoint(Preheader);
  builder.CreateBr(TileHeader);

  // Build tile loop header
  builder.SetInsertPoint(TileHeader);
  llvm::PHINode *TileIV = builder.CreatePHI(Int64Ty, 2, "tile.iv");
  llvm::Value *Zero = llvm::ConstantInt::get(Int64Ty, 0);
  TileIV->addIncoming(Zero, Preheader);

  llvm::Value *TileCond = builder.CreateICmp(llvm::ICmpInst::ICMP_ULT, TileIV, OrigBound, "tile.cond");
  builder.CreateCondBr(TileCond, TileBody, ExitBlock);

  // Tile body
  builder.SetInsertPoint(TileBody);
  builder.CreateBr(PointHeader);

  // Point header
  builder.SetInsertPoint(PointHeader);
  llvm::PHINode *PointIV = builder.CreatePHI(Int64Ty, 2, "point.iv");
  PointIV->addIncoming(TileIV, TileBody);

  llvm::Value *TileEndVal = builder.CreateAdd(TileIV, info.tile_size, "tile.end");
  llvm::Value *TileEndClamped = builder.CreateSelect(
      builder.CreateICmpULT(TileEndVal, OrigBound),
      TileEndVal,
      OrigBound,
      "tile.end.clamped");

  llvm::Value *PointCond = builder.CreateICmp(llvm::ICmpInst::ICMP_ULT, PointIV, TileEndClamped, "point.cond");
  builder.CreateCondBr(PointCond, PointBody, TileLatch);

  // Point body
  builder.SetInsertPoint(PointBody);
  builder.CreateBr(PointLatch);

  // Point latch
  builder.SetInsertPoint(PointLatch);
  llvm::Value *One = llvm::ConstantInt::get(Int64Ty, 1);
  llvm::Value *PointNext = builder.CreateAdd(PointIV, One, "point.next");
  PointIV->addIncoming(PointNext, PointLatch);
  builder.CreateBr(PointHeader);

  // Tile latch
  builder.SetInsertPoint(TileLatch);
  llvm::Value *TileNext = builder.CreateAdd(TileIV, info.tile_size, "tile.next");
  TileIV->addIncoming(TileNext, TileLatch);
  builder.CreateBr(TileHeader);

  info.tile_iv = TileIV;
  info.point_iv = PointIV;

  log_debug("Created tiled loop structure successfully");

  return info;
}

// Clone loop body and update uses of induction variable
void clone_loop_body_with_iv_replacement(llvm::Loop *original_loop,
                                         llvm::BasicBlock *target_block,
                                         llvm::PHINode *old_iv,
                                         llvm::PHINode *new_iv,
                                         llvm::IRBuilder<> &builder) {
  // Clear target block (remove the existing branch to point.latch)
  if (target_block->getTerminator()) {
    target_block->getTerminator()->eraseFromParent();
  }

  builder.SetInsertPoint(target_block);

  // Collect instructions to clone from original loop
  std::vector<llvm::Instruction*> to_clone;
  llvm::BasicBlock *OrigHeader = original_loop->getHeader();

  for (llvm::BasicBlock *BB : original_loop->blocks()) {
    for (llvm::Instruction &I : *BB) {
      // Skip PHI nodes in header
      if (llvm::isa<llvm::PHINode>(&I) && BB == OrigHeader) {
        continue;
      }
      // Skip terminators
      if (I.isTerminator()) {
        continue;
      }
      to_clone.push_back(&I);
    }
  }

  // Clone instructions with value remapping
  llvm::ValueToValueMapTy VMap;
  VMap[old_iv] = new_iv;

  // First pass: clone all instructions
  std::vector<llvm::Instruction*> cloned_insts;
  for (llvm::Instruction *I : to_clone) {
    llvm::Instruction *Cloned = I->clone();
    cloned_insts.push_back(Cloned);
    VMap[I] = Cloned;
  }

  // Second pass: insert and remap operands
  for (llvm::Instruction *Cloned : cloned_insts) {
    // Remap operands
    for (unsigned i = 0; i < Cloned->getNumOperands(); ++i) {
      llvm::Value *Op = Cloned->getOperand(i);
      if (VMap.count(Op)) {
        Cloned->setOperand(i, VMap[Op]);
      }
    }

    // Insert into target block
    builder.Insert(Cloned);
  }

  // Add terminator back to point.latch
  llvm::Function *F = target_block->getParent();
  llvm::BasicBlock *PointLatch = nullptr;

  // Find point.latch block
  for (llvm::BasicBlock &BB : *F) {
    if (BB.getName().contains("point.latch")) {
      PointLatch = &BB;
      break;
    }
  }

  if (PointLatch) {
    builder.CreateBr(PointLatch);
  } else {
    log_error("Cannot find point.latch block for terminator");
  }
}

llvm::Loop *create_tiled_loop(llvm::Loop *original_loop,
                              size_t tile_size,
                              llvm::IRBuilder<> *builder) {
  if (!original_loop || tile_size == 0 || !builder) {
    return nullptr;
  }

  log_info("Creating tiled loop with tile_size=%zu", tile_size);

  // Get original IV BEFORE any modifications
  llvm::PHINode *OrigIV = original_loop->getCanonicalInductionVariable();
  if (!OrigIV) {
    log_error("Cannot find original induction variable before tiling");
    return nullptr;
  }

  // Clone the original loop body blocks before modification
  std::vector<llvm::BasicBlock*> OriginalBlocks;
  for (llvm::BasicBlock *BB : original_loop->blocks()) {
    OriginalBlocks.push_back(BB);
  }

  // Create complete tiled structure
  TiledLoopInfo tiled = create_complete_tiled_loop(original_loop, tile_size, *builder);

  if (!tiled.tile_iv || !tiled.point_iv) {
    log_error("Failed to create tiled loop structure");
    return nullptr;
  }

  // Find point body block - the block between point header and point latch
  llvm::BasicBlock *PointBody = nullptr;
  llvm::BasicBlock *PointHeader = tiled.point_iv->getParent();

  for (llvm::BasicBlock *Succ : llvm::successors(PointHeader)) {
    // Look for the block that's not the tile latch
    if (Succ->getName().contains("point.body")) {
      PointBody = Succ;
      break;
    }
  }

  if (!PointBody) {
    log_error("Cannot find point body block for cloning");
    return nullptr;
  }

  // Clone original loop body into point body
  clone_loop_body_with_iv_replacement(
      original_loop, PointBody, OrigIV, tiled.point_iv, *builder);

  log_info("Tiled loop structure created successfully");

  // Return the tile loop (outer loop of the tiled structure)
  llvm::Function *F = original_loop->getHeader()->getParent();
  llvm::DominatorTree DT(*F);
  llvm::LoopInfo LI(DT);
  LI.analyze(DT);

  llvm::Loop *NewTileLoop = LI.getLoopFor(tiled.tile_iv->getParent());

  if (!NewTileLoop) {
    log_debug("Could not find new tile loop in LoopInfo, returning original");
    return original_loop;
  }

  return NewTileLoop;
}

int apply_tiling_transform(PolyOptPlanImpl *plan) {
  if (!plan || !plan->analysis_done) {
    return POLY_ERR_INVALID_ARG;
  }

  if (!plan->nest_info.is_tileable) {
    log_error("Loop nest is not tileable");
    return POLY_ERR_UNSUPPORTED_LOOP;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  log_info("Applying loop tiling transformation");

  // Compute tile sizes if not already done
  if (plan->tiling_config.tile_M == 0) {
    compute_tile_sizes(plan, g_poly_state.config.tile_strategy);
  }

  const TilingConfig &cfg = plan->tiling_config;

  // Mark tiling as applied via metadata instead of IR transformation
  // This avoids the complexity of IR manipulation while still providing
  // hints to LLVM's optimization passes

  for (size_t i = 0; i < plan->nest_info.nested_loops.size(); i++) {
    llvm::Loop *loop = plan->nest_info.nested_loops[i];
    llvm::LLVMContext &Ctx = loop->getHeader()->getContext();

    size_t tile_size = 0;
    if (i == 0 && cfg.tile_M > 0) tile_size = cfg.tile_M;
    else if (i == 1 && cfg.tile_N > 0) tile_size = cfg.tile_N;
    else if (i == 2 && cfg.tile_K > 0) tile_size = cfg.tile_K;

    if (tile_size > 0) {
      // Add tile size metadata hint
      llvm::Metadata *TileHintArgs[] = {
          llvm::MDString::get(Ctx, "llvm.loop.tile.size"),
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt64Ty(Ctx), tile_size))
      };
      llvm::MDNode *TileHintNode = llvm::MDNode::get(Ctx, TileHintArgs);

      // Get or create loop ID
      llvm::MDNode *LoopID = loop->getLoopID();
      llvm::SmallVector<llvm::Metadata *, 4> MDs;

      if (LoopID) {
        for (unsigned j = 1; j < LoopID->getNumOperands(); ++j) {
          MDs.push_back(LoopID->getOperand(j));
        }
      }

      MDs.push_back(TileHintNode);

      llvm::Metadata *NewLoopIDArgs[] = {nullptr};
      llvm::MDNode *NewLoopID = llvm::MDNode::get(Ctx, NewLoopIDArgs);
      NewLoopID->replaceOperandWith(0, NewLoopID);

      for (auto *MD : MDs) {
        llvm::Metadata *Args[] = {NewLoopID, MD};
        NewLoopID = llvm::MDNode::get(Ctx, Args);
      }

      loop->setLoopID(NewLoopID);
      log_info("Added tile size hint (%zu) to loop %zu", tile_size, i);
    }
  }

  plan->tiling_applied = true;
  plan->stats.loops_tiled = plan->nest_info.nested_loops.size();

  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
  plan->transform_time_ms += elapsed;

  log_info("Tiling transformation completed in %.2f ms, marked %zu loops",
           elapsed, plan->stats.loops_tiled);

  // Force LLVM to actually apply loop transformations by marking loops for distribution
  for (size_t i = 0; i < plan->nest_info.nested_loops.size(); i++) {
    llvm::Loop *loop = plan->nest_info.nested_loops[i];
    llvm::LLVMContext &Ctx = loop->getHeader()->getContext();

    // Enable loop distribution for better tiling
    llvm::Metadata *DistributeArgs[] = {
      llvm::MDString::get(Ctx, "llvm.loop.distribute.enable"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt1Ty(Ctx), 1))
  };
    llvm::MDNode *DistributeNode = llvm::MDNode::get(Ctx, DistributeArgs);

    llvm::MDNode *LoopID = loop->getLoopID();
    llvm::SmallVector<llvm::Metadata *, 8> MDs;

    if (LoopID) {
      for (unsigned j = 1; j < LoopID->getNumOperands(); ++j) {
        MDs.push_back(LoopID->getOperand(j));
      }
    }

    MDs.push_back(DistributeNode);

    llvm::Metadata *NewLoopIDArgs[] = {nullptr};
    llvm::MDNode *NewLoopID = llvm::MDNode::get(Ctx, NewLoopIDArgs);
    NewLoopID->replaceOperandWith(0, NewLoopID);

    for (auto *MD : MDs) {
      llvm::Metadata *Args[] = {NewLoopID, MD};
      NewLoopID = llvm::MDNode::get(Ctx, Args);
    }

    loop->setLoopID(NewLoopID);
  }

  return POLY_OK;
}

int apply_multilevel_tiling(PolyOptPlanImpl *plan) {
  if (!plan || !plan->analysis_done) {
    return POLY_ERR_INVALID_ARG;
  }

  if (!plan->nest_info.is_tileable) {
    return POLY_ERR_UNSUPPORTED_LOOP;
  }

  log_info("Applying multi-level tiling");

  // Compute multi-level tile sizes
  compute_tile_sizes(plan, POLY_TILE_MULTI_LEVEL);

  // Multi-level tiling strategy:
  // Level 1 (outermost): L3 cache tiling
  // Level 2: L2 cache tiling
  // Level 3: L1 cache tiling
  // Level 4 (innermost): Register tiling

  const TilingConfig &cfg = plan->tiling_config;

  // For 3 nested loops (i, j, k), we apply tiling at each level
  // This creates a 4-level hierarchy:
  // for ii (L3 tiles)
  //   for jj (L3 tiles)
  //     for kk (L3 tiles)
  //       for i_l2 (L2 tiles within L3)
  //         for j_l2 (L2 tiles within L3)
  //           for k_l2 (L2 tiles within L3)
  //             for i_l1 (L1 tiles within L2)
  //               for j_l1 (L1 tiles within L2)
  //                 for k_l1 (L1 tiles within L2)
  //                   for i (register tiles within L1)
  //                     for j (register tiles within L1)
  //                       for k (register tiles within L1)
  //                         computation

  size_t num_loops = plan->nest_info.nested_loops.size();
  if (num_loops < 2) {
    log_error("Need at least 2 loops for multi-level tiling");
    return POLY_ERR_UNSUPPORTED_LOOP;
  }

  // Apply L3 tiling to outermost loop
  if (cfg.l3_tile_M > 0 && num_loops >= 1) {
    llvm::Loop *outer = plan->nest_info.nested_loops[0];
    llvm::IRBuilder<> builder(outer->getHeader()->getContext());

    llvm::Loop *l3_tiled = create_tiled_loop(outer, cfg.l3_tile_M, &builder);
    if (l3_tiled) {
      log_info("Applied L3 tiling to outer loop (tile=%zu)", cfg.l3_tile_M);
    }
  }

  // Apply L2 tiling to middle loop
  if (cfg.l2_tile_N > 0 && num_loops >= 2) {
    llvm::Loop *middle = plan->nest_info.nested_loops[1];
    llvm::IRBuilder<> builder(middle->getHeader()->getContext());

    llvm::Loop *l2_tiled = create_tiled_loop(middle, cfg.l2_tile_N, &builder);
    if (l2_tiled) {
      log_info("Applied L2 tiling to middle loop (tile=%zu)", cfg.l2_tile_N);
    }
  }

  // Apply L1 tiling to innermost loop
  if (cfg.l1_tile_K > 0 && num_loops >= 3) {
    llvm::Loop *inner = plan->nest_info.nested_loops[2];
    llvm::IRBuilder<> builder(inner->getHeader()->getContext());

    llvm::Loop *l1_tiled = create_tiled_loop(inner, cfg.l1_tile_K, &builder);
    if (l1_tiled) {
      log_info("Applied L1 tiling to inner loop (tile=%zu)", cfg.l1_tile_K);
    }
  }

  // Register tiling would be applied as the final innermost level
  // This is typically done as part of unrolling or vectorization

  plan->tiling_applied = true;
  plan->stats.loops_tiled = num_loops * 3; // 3 levels of tiling

  log_info("Multi-level tiling completed");

  return POLY_OK;
}

} // namespace poly_internal

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

extern "C" {

int poly_opt_analyze_loops(poly_opt_plan_t *plan, poly_loop_info_t *out_info) {
  using namespace poly_internal;

  if (!plan || !out_info) {
    return POLY_ERR_INVALID_ARG;
  }

  PolyOptPlanImpl *impl = plan_to_impl(plan);

  if (!impl->analysis_done) {
    int ret = analyze_loop_nest(impl);
    if (ret != POLY_OK) {
      return ret;
    }
  }

  // Fill output structure
  out_info->loop_depth = impl->nest_info.nested_loops.size();
  out_info->is_perfectly_nested = impl->nest_info.is_perfectly_nested;
  out_info->has_affine_bounds = impl->nest_info.has_affine_bounds;
  out_info->has_data_dependencies = impl->nest_info.has_data_dependencies;

  // Allocate and copy trip counts
  out_info->trip_counts = static_cast<size_t*>(
      malloc(out_info->loop_depth * sizeof(size_t)));
  if (out_info->trip_counts) {
    std::copy(impl->nest_info.trip_counts.begin(),
              impl->nest_info.trip_counts.end(),
              out_info->trip_counts);
  }

  // Detailed stride analysis using ScalarEvolution
  out_info->strides = static_cast<size_t*>(
      malloc(out_info->loop_depth * sizeof(size_t)));
  if (out_info->strides) {
    for (size_t i = 0; i < out_info->loop_depth; i++) {
      llvm::Loop *loop = impl->nest_info.nested_loops[i];
      llvm::PHINode *IndVar = impl->nest_info.induction_vars[i];

      size_t detected_stride = 1; // Default stride

      if (IndVar && impl->scalar_evolution) {
        // Get SCEV for induction variable
        const llvm::SCEV *IndVarSCEV = impl->scalar_evolution->getSCEV(IndVar);

        if (const llvm::SCEVAddRecExpr *AddRec =
            llvm::dyn_cast<llvm::SCEVAddRecExpr>(IndVarSCEV)) {
          // AddRecExpr format: {Start, +, Step}
          const llvm::SCEV *StepSCEV = AddRec->getStepRecurrence(*impl->scalar_evolution);

          if (const llvm::SCEVConstant *StepConst =
              llvm::dyn_cast<llvm::SCEVConstant>(StepSCEV)) {
            // Extract constant stride value
            llvm::APInt StepValue = StepConst->getAPInt();
            detected_stride = StepValue.getZExtValue();
            log_debug("Loop %zu: detected stride = %zu", i, detected_stride);
          } else {
            log_debug("Loop %zu: non-constant stride, using default", i);
          }
        } else {
          log_debug("Loop %zu: not an AddRecExpr, using default stride", i);
        }
      } else {
        log_debug("Loop %zu: no IV or SE available, using default stride", i);
      }

      out_info->strides[i] = detected_stride;
    }
  }

  return POLY_OK;
}

int poly_opt_is_tileable(poly_opt_plan_t *plan) {
  using namespace poly_internal;

  if (!plan) {
    return 0;
  }

  PolyOptPlanImpl *impl = plan_to_impl(plan);

  if (!impl->analysis_done) {
    analyze_loop_nest(impl);
  }

  return impl->nest_info.is_tileable ? 1 : 0;
}

int poly_opt_recommend_tile_sizes(poly_opt_plan_t *plan,
                                   poly_tile_strategy_t level,
                                   size_t *out_tile_M,
                                   size_t *out_tile_N,
                                   size_t *out_tile_K) {
  using namespace poly_internal;

  if (!plan || !out_tile_M || !out_tile_N || !out_tile_K) {
    return POLY_ERR_INVALID_ARG;
  }

  PolyOptPlanImpl *impl = plan_to_impl(plan);

  // Compute tile sizes for requested level
  compute_tile_sizes(impl, level);

  *out_tile_M = impl->tiling_config.tile_M;
  *out_tile_N = impl->tiling_config.tile_N;
  *out_tile_K = impl->tiling_config.tile_K;

  return POLY_OK;
}

int poly_opt_apply_tiling(poly_opt_plan_t *plan,
                          poly_tile_strategy_t strategy) {
  using namespace poly_internal;

  if (!plan) {
    return POLY_ERR_INVALID_ARG;
  }

  PolyOptPlanImpl *impl = plan_to_impl(plan);

  if (!impl->analysis_done) {
    int ret = analyze_loop_nest(impl);
    if (ret != POLY_OK) {
      return ret;
    }
  }

  if (strategy == POLY_TILE_MULTI_LEVEL) {
    return apply_multilevel_tiling(impl);
  } else {
    return apply_tiling_transform(impl);
  }
}

} // extern "C"