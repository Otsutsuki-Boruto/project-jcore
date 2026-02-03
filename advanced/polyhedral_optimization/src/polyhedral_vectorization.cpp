// advanced/polyhedral_optimization/src/polyhedral_vectorization.cpp

/**
 * @file polyhedral_vectorization.cpp
 * @brief Loop vectorization preparation and unrolling transformations
 *
 * Prepares loops for vectorization by inserting hints and metadata,
 * and applies loop unrolling optimizations.
 */

#include "polyhedral_internal.h"
#include <llvm/Transforms/Utils/LoopUtils.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>

namespace poly_internal {

/* ========================================================================== */
/* Vectorization Analysis */
/* ========================================================================== */

bool is_vectorizable(llvm::Loop *loop) {
  if (!loop) return false;

  // Check basic requirements for vectorization

  // 1. Loop must have a single latch
  if (!loop->getLoopLatch()) {
    log_debug("Loop not vectorizable: no single latch");
    return false;
  }

  // 2. Loop must have preheader
  if (!loop->getLoopPreheader()) {
    log_debug("Loop not vectorizable: no preheader");
    return false;
  }

  // 3. Check for canonical induction variable
  llvm::PHINode *IndVar = loop->getCanonicalInductionVariable();
  if (!IndVar) {
    log_debug("Loop not vectorizable: no canonical IV");
    return false;
  }

  // 4. Check for simple memory access patterns
  bool has_complex_control_flow = false;
  for (llvm::BasicBlock *BB : loop->blocks()) {
    // Count successors - if > 2, complex control flow
    if (BB->getTerminator()->getNumSuccessors() > 2) {
      has_complex_control_flow = true;
      break;
    }
  }

  if (has_complex_control_flow) {
    log_debug("Loop not vectorizable: complex control flow");
    return false;
  }

  // 5. Check for function calls that might not be vectorizable
  for (llvm::BasicBlock *BB : loop->blocks()) {
    for (llvm::Instruction &I : *BB) {
      if (llvm::isa<llvm::CallInst>(&I)) {
        llvm::CallInst *CI = llvm::cast<llvm::CallInst>(&I);
        // Check if it's an intrinsic (usually safe to vectorize)
        if (!CI->getCalledFunction() ||
            !CI->getCalledFunction()->isIntrinsic()) {
          log_debug("Loop not vectorizable: contains non-intrinsic call");
          return false;
        }
      }
    }
  }

  log_debug("Loop appears vectorizable");
  return true;
}

/* ========================================================================== */
/* Vectorization Metadata */
/* ========================================================================== */

void insert_vectorization_hints(llvm::Loop *loop) {
  if (!loop) return;

  llvm::BasicBlock *Latch = loop->getLoopLatch();
  if (!Latch) return;

  llvm::BranchInst *BI = llvm::dyn_cast<llvm::BranchInst>(
      Latch->getTerminator());
  if (!BI) return;

  // Create metadata for vectorization
  llvm::LLVMContext &Ctx = loop->getHeader()->getContext();

  // Create "llvm.loop.vectorize.enable" metadata
  llvm::Metadata *EnableArgs[] = {
      llvm::MDString::get(Ctx, "llvm.loop.vectorize.enable"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt1Ty(Ctx), 1))
  };
  llvm::MDNode *EnableNode = llvm::MDNode::get(Ctx, EnableArgs);

  // Create "llvm.loop.vectorize.width" metadata (hint for vector width)
  llvm::Metadata *WidthArgs[] = {
      llvm::MDString::get(Ctx, "llvm.loop.vectorize.width"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 4))
  };
  llvm::MDNode *WidthNode = llvm::MDNode::get(Ctx, WidthArgs);

  // Create "llvm.loop.interleave.count" metadata
  llvm::Metadata *InterleaveArgs[] = {
      llvm::MDString::get(Ctx, "llvm.loop.interleave.count"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 2))
  };
  llvm::MDNode *InterleaveNode = llvm::MDNode::get(Ctx, InterleaveArgs);

  // Create root loop metadata node
  llvm::Metadata *LoopMDArgs[] = {
      nullptr, // Self-reference (filled below)
      EnableNode,
      WidthNode,
      InterleaveNode
  };
  llvm::MDNode *LoopMD = llvm::MDNode::get(Ctx, LoopMDArgs);
  LoopMD->replaceOperandWith(0, LoopMD); // Self-reference

  // Attach metadata to loop latch branch
  BI->setMetadata(llvm::LLVMContext::MD_loop, LoopMD);

  log_debug("Inserted vectorization hints for loop");
}

/* ========================================================================== */
/* Vectorization Preparation */
/* ========================================================================== */

int prepare_for_vectorization(PolyOptPlanImpl *plan) {
  if (!plan || !plan->analysis_done) {
    return POLY_ERR_INVALID_ARG;
  }

  log_info("Preparing loops for vectorization");

  size_t vectorizable_count = 0;

  for (llvm::Loop *loop : plan->nest_info.nested_loops) {
    if (is_vectorizable(loop)) {
      insert_vectorization_hints(loop);
      vectorizable_count++;
    }
  }

  plan->vectorization_applied = true;
  plan->stats.loops_vectorized = vectorizable_count;

  log_info("Prepared %zu loops for vectorization", vectorizable_count);

  return POLY_OK;
}

/* ========================================================================== */
/* Loop Unrolling */
/* ========================================================================== */

int compute_unroll_factor(llvm::Loop *loop, size_t trip_count) {
  if (!loop || trip_count == 0) {
    return 1; // No unrolling
  }

  // Heuristics for unroll factor selection

  // For very small trip counts, unroll completely
  if (trip_count <= 4) {
    return static_cast<int>(trip_count);
  }

  // For small trip counts, unroll by 2-4
  if (trip_count <= 16) {
    return 4;
  }

  // For medium trip counts, unroll by 4-8
  if (trip_count <= 64) {
    return 8;
  }

  // For large trip counts, moderate unrolling
  if (trip_count <= 256) {
    return 8;
  }

  // For very large trip counts, minimal unrolling
  return 4;
}

int apply_unrolling_transform(PolyOptPlanImpl *plan,
                               int outer_factor,
                               int inner_factor) {
  if (!plan || !plan->analysis_done) {
    return POLY_ERR_INVALID_ARG;
  }

  log_info("Applying loop unrolling: outer=%d, inner=%d",
           outer_factor, inner_factor);

  size_t num_loops = plan->nest_info.nested_loops.size();
  if (num_loops == 0) {
    log_error("No loops to unroll");
    return POLY_ERR_UNSUPPORTED_LOOP;
  }

  // Auto-compute factors if not specified
  if (outer_factor == 0 || inner_factor == 0) {
    if (num_loops >= 2) {
      if (outer_factor == 0) {
        size_t outer_trip = plan->nest_info.trip_counts[0];
        outer_factor = compute_unroll_factor(
            plan->nest_info.nested_loops[0], outer_trip);
      }
      if (inner_factor == 0) {
        size_t inner_trip = plan->nest_info.trip_counts[num_loops - 1];
        inner_factor = compute_unroll_factor(
            plan->nest_info.nested_loops[num_loops - 1], inner_trip);
      }
    } else if (num_loops == 1) {
      if (inner_factor == 0) {
        size_t trip = plan->nest_info.trip_counts[0];
        inner_factor = compute_unroll_factor(
            plan->nest_info.nested_loops[0], trip);
      }
    }
  }

  log_info("Computed unroll factors: outer=%d, inner=%d",
           outer_factor, inner_factor);

  // Apply actual LLVM loop unrolling using metadata
  llvm::LLVMContext &Ctx = plan->nest_info.nested_loops[0]->getHeader()->getContext();

  // Unroll inner loop(s)
  if (inner_factor > 1 && num_loops > 0) {
    llvm::Loop *innerLoop = plan->nest_info.nested_loops[num_loops - 1];

    // Create unroll metadata
    llvm::Metadata *UnrollCountArgs[] = {
        llvm::MDString::get(Ctx, "llvm.loop.unroll.count"),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), inner_factor))
    };
    llvm::MDNode *UnrollCountNode = llvm::MDNode::get(Ctx, UnrollCountArgs);

    llvm::Metadata *UnrollEnableArgs[] = {
        llvm::MDString::get(Ctx, "llvm.loop.unroll.enable")
    };
    llvm::MDNode *UnrollEnableNode = llvm::MDNode::get(Ctx, UnrollEnableArgs);

    // Get existing loop ID or create new one
    llvm::MDNode *LoopID = innerLoop->getLoopID();
    llvm::SmallVector<llvm::Metadata *, 4> MDs;

    if (LoopID) {
      for (unsigned i = 1; i < LoopID->getNumOperands(); ++i) {
        MDs.push_back(LoopID->getOperand(i));
      }
    }

    MDs.push_back(UnrollCountNode);
    MDs.push_back(UnrollEnableNode);

    llvm::Metadata *NewLoopIDArgs[] = {nullptr};
    llvm::MDNode *NewLoopID = llvm::MDNode::get(Ctx, NewLoopIDArgs);
    NewLoopID->replaceOperandWith(0, NewLoopID);

    for (auto *MD : MDs) {
      llvm::Metadata *Args[] = {NewLoopID, MD};
      NewLoopID = llvm::MDNode::get(Ctx, Args);
    }

    innerLoop->setLoopID(NewLoopID);
    log_debug("Inner loop marked for unrolling with factor %d", inner_factor);
  }

  // Unroll outer loop if applicable
  if (outer_factor > 1 && num_loops >= 2) {
    llvm::Loop *outerLoop = plan->nest_info.nested_loops[0];

    llvm::Metadata *UnrollCountArgs[] = {
        llvm::MDString::get(Ctx, "llvm.loop.unroll.count"),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), outer_factor))
    };
    llvm::MDNode *UnrollCountNode = llvm::MDNode::get(Ctx, UnrollCountArgs);

    llvm::Metadata *UnrollEnableArgs[] = {
        llvm::MDString::get(Ctx, "llvm.loop.unroll.enable")
    };
    llvm::MDNode *UnrollEnableNode = llvm::MDNode::get(Ctx, UnrollEnableArgs);

    llvm::MDNode *LoopID = outerLoop->getLoopID();
    llvm::SmallVector<llvm::Metadata *, 4> MDs;

    if (LoopID) {
      for (unsigned i = 1; i < LoopID->getNumOperands(); ++i) {
        MDs.push_back(LoopID->getOperand(i));
      }
    }

    MDs.push_back(UnrollCountNode);
    MDs.push_back(UnrollEnableNode);

    llvm::Metadata *NewLoopIDArgs[] = {nullptr};
    llvm::MDNode *NewLoopID = llvm::MDNode::get(Ctx, NewLoopIDArgs);
    NewLoopID->replaceOperandWith(0, NewLoopID);

    for (auto *MD : MDs) {
      llvm::Metadata *Args[] = {NewLoopID, MD};
      NewLoopID = llvm::MDNode::get(Ctx, Args);
    }

    outerLoop->setLoopID(NewLoopID);
    log_debug("Outer loop marked for unrolling with factor %d", outer_factor);
  }

  plan->unrolling_applied = true;

  return POLY_OK;
}

/* ========================================================================== */
/* Loop Peeling */
/* ========================================================================== */

int apply_loop_peeling(llvm::Loop *loop, size_t peel_count) {
    if (!loop || peel_count == 0) {
        return POLY_OK;
    }

    log_debug("Peeling loop with %zu iterations", peel_count);

    llvm::BasicBlock *preheader = loop->getLoopPreheader();
    llvm::BasicBlock *header = loop->getHeader();

    if (!preheader || !header) {
        log_error("Loop has no preheader/header; cannot peel");
        return POLY_ERR_UNSUPPORTED_LOOP;
    }

    // Get the loop induction variable
    llvm::PHINode *indVar = loop->getCanonicalInductionVariable();
    if (!indVar) {
        log_error("Loop has no canonical induction variable; cannot peel");
        return POLY_ERR_UNSUPPORTED_LOOP;
    }

    llvm::IRBuilder<> builder(preheader->getTerminator());

    // Compute the peeled iteration values
    llvm::SmallVector<llvm::Value*, 8> peeledValues;
    for (size_t i = 0; i < peel_count; ++i) {
        peeledValues.push_back(llvm::ConstantInt::get(indVar->getType(), i));
    }

    // Clone loop body for each peeled iteration
    for (size_t i = 0; i < peel_count; ++i) {
        // Map for cloning
        llvm::ValueToValueMapTy VMap;

        for (auto &inst : *header) {
          llvm::Instruction *clonedInst = inst.clone();
            VMap[&inst] = clonedInst;

            // Replace induction variable with constant for this peeled iteration
            if (&inst == indVar) {
                VMap[&inst] = peeledValues[i];
            }
        }
      // Get iterator to first instruction of the header
      auto insertPos = header->begin();

      for (auto instPair : VMap) {
        if (llvm::Instruction *I = llvm::dyn_cast<llvm::Instruction>(instPair.second)) {
          // Insert using iterator version (non-deprecated)
          I->insertBefore(insertPos);
        }
      }
    }

    // Adjust the original induction variable start
  auto startVal = llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(indVar->getType(), peel_count));
  indVar->setIncomingValueForBlock(preheader, startVal);

    log_debug("Loop peeling applied successfully for %zu iterations", peel_count);
    return POLY_OK;
}


/* ========================================================================== */
/* Prefetch Insertion */
/* ========================================================================== */

void insert_prefetch_hints(llvm::Loop *loop) {
  if (!loop) return;

  // Insert prefetch intrinsics for memory accesses
  llvm::Function *F = loop->getHeader()->getParent();
  if (!F) return;

  llvm::Module *M = F->getParent();
  if (!M) return;

  llvm::LLVMContext &Ctx = M->getContext();

  // Get prefetch intrinsic
  llvm::Function *PrefetchFn = llvm::Intrinsic::getOrInsertDeclaration(
      M, llvm::Intrinsic::prefetch, {llvm::PointerType::get(llvm::Type::getInt8Ty(Ctx), 0)});

  if (!PrefetchFn) {
    log_debug("Prefetch intrinsic not available");
    return;
  }

  // Find memory accesses in loop
  for (llvm::BasicBlock *BB : loop->blocks()) {
    if (!BB) continue;

    for (llvm::Instruction &I : *BB) {
      if (llvm::LoadInst *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        // Insert prefetch before load
        llvm::IRBuilder<> Builder(LI);

        llvm::Value *Ptr = LI->getPointerOperand();
        if (!Ptr) continue;

        // Cast pointer to i8* for prefetch intrinsic
        llvm::Value *PtrCast = Builder.CreateBitCast(
            Ptr, llvm::PointerType::get(llvm::Type::getInt8Ty(Ctx), 0));

        // Prefetch arguments: (ptr, rw, locality, cache_type)
        llvm::Value *Args[] = {
            PtrCast,
            Builder.getInt32(0), // 0=read, 1=write
            Builder.getInt32(3), // locality: 0=none, 3=high
            Builder.getInt32(1)  // cache type: 1=data
        };

        Builder.CreateCall(PrefetchFn, Args);

        log_debug("Inserted prefetch for load instruction");
      }
    }
  }
}

int apply_prefetching(PolyOptPlanImpl *plan) {
  if (!plan || !plan->analysis_done) {
    return POLY_ERR_INVALID_ARG;
  }

  if (!g_poly_state.config.enable_prefetch) {
    return POLY_OK;
  }

  log_info("Inserting prefetch hints");

  for (llvm::Loop *loop : plan->nest_info.nested_loops) {
    insert_prefetch_hints(loop);
  }

  return POLY_OK;
}

} // namespace poly_internal

/* ========================================================================== */
/* Public API Implementation */
/* ========================================================================== */

extern "C" {

int poly_opt_apply_vectorization(poly_opt_plan_t *plan) {
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

  int ret = prepare_for_vectorization(impl);
  if (ret != POLY_OK) {
    return ret;
  }

  // Also apply prefetching if enabled
  if (g_poly_state.config.enable_prefetch) {
    apply_prefetching(impl);
  }

  return POLY_OK;
}

int poly_opt_apply_unrolling(poly_opt_plan_t *plan,
                              int outer_factor,
                              int inner_factor) {
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

  return apply_unrolling_transform(impl, outer_factor, inner_factor);
}

} // extern "C"