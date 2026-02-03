// advanced/jcore_neuralPrimitives/src/NPL_tensor_ops.cpp

#include <llvm/IR/Module.h>

#include "neural_primitives_internal.h"
#include "jit_kernel_generator.h"
#include "ffm_prefetch.h"
#include "polyhedral_optimization.h"

using namespace npl_internal;

/* ========================================================================== */
/* Tensor Management                                                           */
/* ========================================================================== */

int npl_create_tensor(void *data, size_t ndim, const size_t *shape,
                       npl_dtype_t dtype, npl_data_layout_t layout,
                       npl_tensor_t *out_tensor) {
  if (!out_tensor || !shape || ndim == 0 || ndim > NPL_MAX_DIMS) {
    return NPL_ERR_INVALID_ARG;
  }

  out_tensor->data = data;
  out_tensor->ndim = ndim;
  out_tensor->dtype = dtype;
  out_tensor->layout = layout;

  // Copy shape
  size_t total_elements = 1;
  for (size_t i = 0; i < ndim; ++i) {
    out_tensor->shape[i] = shape[i];
    total_elements *= shape[i];
  }

  // Compute strides (row-major by default)
  out_tensor->strides[ndim - 1] = 1;
  for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
    out_tensor->strides[i] = out_tensor->strides[i + 1] * shape[i + 1];
  }

  // Compute total size
  size_t elem_size = GetElementSize(dtype);
  out_tensor->size_bytes = total_elements * elem_size;
  out_tensor->is_contiguous = 1;

  return NPL_OK;
}

int npl_allocate_tensor(npl_tensor_t *tensor) {
  if (!tensor) {
    return NPL_ERR_INVALID_ARG;
  }

  if (tensor->data) {
    return NPL_OK; // Already allocated
  }

  if (tensor->size_bytes == 0) {
    return NPL_ERR_INVALID_ARG;
  }

  // Allocate aligned memory
  tensor->data = AllocateAligned(tensor->size_bytes, 64);
  if (!tensor->data) {
    return NPL_ERR_NO_MEMORY;
  }

  return NPL_OK;
}

void npl_free_tensor(npl_tensor_t *tensor) {
  if (tensor && tensor->data) {
    FreeAligned(tensor->data);
    tensor->data = nullptr;
  }
}

int npl_reshape_tensor(const npl_tensor_t *tensor, size_t new_ndim,
                        const size_t *new_shape, npl_tensor_t *out_tensor) {
  if (!tensor || !new_shape || !out_tensor || new_ndim > NPL_MAX_DIMS) {
    return NPL_ERR_INVALID_ARG;
  }

  // Verify total elements remain the same
  size_t old_elements = GetTensorElementCount(tensor);
  size_t new_elements = 1;
  for (size_t i = 0; i < new_ndim; ++i) {
    new_elements *= new_shape[i];
  }

  if (old_elements != new_elements) {
    return NPL_ERR_SHAPE_MISMATCH;
  }

  // Create new tensor descriptor
  *out_tensor = *tensor; // Copy everything
  out_tensor->ndim = new_ndim;
  for (size_t i = 0; i < new_ndim; ++i) {
    out_tensor->shape[i] = new_shape[i];
  }

  // Recompute strides
  out_tensor->strides[new_ndim - 1] = 1;
  for (int i = static_cast<int>(new_ndim) - 2; i >= 0; --i) {
    out_tensor->strides[i] = out_tensor->strides[i + 1] * new_shape[i + 1];
  }

  return NPL_OK;
}

/* ========================================================================== */
/* Matrix Multiplication                                                       */
/* ========================================================================== */

int npl_matmul(const npl_tensor_t *A, const npl_tensor_t *B,
                npl_tensor_t *C, float alpha, float beta,
                npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  // Validate inputs
  int ret = ValidateTensor(A, "A");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(B, "B");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(C, "C");
  if (ret != NPL_OK) return ret;

  if (A->dtype != NPL_DTYPE_FP32 || B->dtype != NPL_DTYPE_FP32 ||
      C->dtype != NPL_DTYPE_FP32) {
    return NPL_ERR_UNSUPPORTED; // Only FP32 for now
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Check if batched matmul
  bool is_batched = (A->ndim >= 3 || B->ndim >= 3);

  if (is_batched) {
    ret = ExecuteMatMulBatched(A, B, C, alpha, beta, stats);
  } else {
    ret = ExecuteMatMulDirect(A, B, C, alpha, beta, stats);
  }

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  g_npl_state.total_ops_executed++;

  return ret;
}

/* ========================================================================== */
/* Element-wise Operations                                                     */
/* ========================================================================== */

int npl_add(const npl_tensor_t *A, const npl_tensor_t *B,
             npl_tensor_t *C, npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(A, "A");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(B, "B");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(C, "C");
  if (ret != NPL_OK) return ret;

  auto start = std::chrono::high_resolution_clock::now();

  ret = ExecuteElementwiseAdd(A, B, C, stats);

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  g_npl_state.total_ops_executed++;

  return ret;
}

int npl_mul(const npl_tensor_t *A, const npl_tensor_t *B,
             npl_tensor_t *C, npl_perf_stats_t *stats) {
  if (!g_npl_state.initialized) {
    return NPL_ERR_NOT_INITIALIZED;
  }

  int ret = ValidateTensor(A, "A");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(B, "B");
  if (ret != NPL_OK) return ret;
  ret = ValidateTensor(C, "C");
  if (ret != NPL_OK) return ret;

  auto start = std::chrono::high_resolution_clock::now();

  ret = ExecuteElementwiseMul(A, B, C, stats);

  if (stats) {
    stats->elapsed_ms = GetElapsedMs(start);
  }

  g_npl_state.total_ops_executed++;

  return ret;
}

/* ========================================================================== */
/* Internal Implementation - MatMul                                            */
/* ========================================================================== */

namespace npl_internal {

int ExecuteMatMulDirect(const npl_tensor_t *A, const npl_tensor_t *B,
                        npl_tensor_t *C, float alpha, float beta,
                        npl_perf_stats_t *stats) {
  // Extract dimensions
  if (A->ndim < 2 || B->ndim < 2 || C->ndim < 2) {
    return NPL_ERR_SHAPE_MISMATCH;
  }

  size_t M = A->shape[A->ndim - 2];
  size_t K = A->shape[A->ndim - 1];
  size_t N = B->shape[B->ndim - 1];

  if (B->shape[B->ndim - 2] != K || C->shape[C->ndim - 2] != M ||
      C->shape[C->ndim - 1] != N) {
    return NPL_ERR_SHAPE_MISMATCH;
      }

  // Prefetch input/output tensors
  ffm_prefetch_block_read_T0(A->data, M * K * sizeof(float));
  ffm_prefetch_block_read_T0(B->data, K * N * sizeof(float));
  ffm_prefetch_block_write_T0(C->data, M * N * sizeof(float));

  // Check if JIT should be used for small kernels (TERM: 2 <= size <= 512)
  bool use_jit = jkg_is_initialized() &&
                 M >= 2 && M <= 512 &&
                 N >= 2 && N <= 512 &&
                 K >= 2 && K <= 512;

  if (use_jit) {
    // Use JIT Kernel Generator
    jkg_kernel_params_t jit_params = {};
    jit_params.M = M;
    jit_params.N = N;
    jit_params.K = K;
    jit_params.alpha = alpha;
    jit_params.beta = beta;
    jit_params.activation = JKG_ACT_NONE;
    jit_params.has_bias = 0;
    jit_params.has_residual = 0;

    jkg_kernel_internal_t *jit_kernel = nullptr;
    int jit_ret = jkg_generate_gemm_tile(M, N, K, &jit_kernel);

    if (jit_ret == JKG_OK && jit_kernel) {
      jkg_gemm_fn gemm_fn = (jkg_gemm_fn)jkg_get_kernel_function(jit_kernel);

      if (gemm_fn) {
        gemm_fn(static_cast<const float *>(A->data),
                static_cast<const float *>(B->data),
                static_cast<float *>(C->data),
                M, N, K, K, N, N, alpha, beta);

        if (stats) {
          double flops = 2.0 * M * N * K;
          stats->gflops = flops / 1e9;
          stats->bandwidth_gbps = 0.0;
          stats->operations_fused = 0;
          stats->memory_saved_bytes = 0;
          stats->backend_used = "JIT-GEMM";
          stats->was_fused = 0;
        }

        jkg_release_kernel(jit_kernel);
        return NPL_OK;
      }

      jkg_release_kernel(jit_kernel);
    }
// JIT failed, fall through to MIL path
  }

  // Use MIL for GEMM (fallback or large kernels)
  mil_perf_stats_t mil_stats = {};
  int ret = mil_sgemm(MIL_LAYOUT_ROW_MAJOR, MIL_NO_TRANS, MIL_NO_TRANS, M, N, K,
                      alpha, static_cast<const float *>(A->data), K,
                      static_cast<const float *>(B->data), N, beta,
                      static_cast<float *>(C->data), N,
                      stats ? &mil_stats : nullptr);

  if (ret != MIL_OK) {
    return NPL_ERR_INTERNAL;
  }

  // Fill statistics
  if (stats) {
    stats->gflops = mil_stats.gflops;
    stats->bandwidth_gbps = mil_stats.bandwidth_gbps;
    stats->operations_fused = 0;
    stats->memory_saved_bytes = 0;
    stats->backend_used = mil_stats.kernel_used;
    stats->was_fused = 0;
  }

  return NPL_OK;
}

int ExecuteMatMulBatched(const npl_tensor_t *A, const npl_tensor_t *B,
                         npl_tensor_t *C, float alpha, float beta,
                         npl_perf_stats_t *stats) {
  // For batched matmul, extract batch dimensions
  size_t batch_size = 1;
  size_t batch_dims = std::max(A->ndim, B->ndim) - 2;

  for (size_t i = 0; i < batch_dims; ++i) {
    batch_size *= A->shape[i];
  }

  size_t M = A->shape[A->ndim - 2];
  size_t K = A->shape[A->ndim - 1];
  size_t N = B->shape[B->ndim - 1];

  size_t stride_A = M * K;
  size_t stride_B = K * N;
  size_t stride_C = M * N;

  const float *A_ptr = static_cast<const float *>(A->data);
  const float *B_ptr = static_cast<const float *>(B->data);
  float *C_ptr = static_cast<float *>(C->data);

  /* ---------------------------------------------------------------------- */
  /* AGEE path: use Adaptive Graph Execution Engine if initialized            */
  /* ---------------------------------------------------------------------- */
  if (agee_is_initialized()) {
    agee_session_t session = nullptr;
    if (agee_create_session(&session) != AGEE_OK) {
      return NPL_ERR_INTERNAL;
    }

    // Build operator graph once (logical batched execution)
    og_graph_t graph = nullptr;
    if (og_create_graph(&graph) != OG_OK) {
      agee_destroy_session(session);
      return NPL_ERR_INTERNAL;
    }

    // Tensors
    uint64_t tid_A, tid_B, tid_C;
    og_tensor_t tA = {};
    og_tensor_t tB = {};
    og_tensor_t tC = {};

    tA.ndim = 2;
    tA.shape[0] = M;
    tA.shape[1] = K;
    tA.data = (void *)A_ptr;

    tB.ndim = 2;
    tB.shape[0] = K;
    tB.shape[1] = N;
    tB.data = (void *)B_ptr;

    tC.ndim = 2;
    tC.shape[0] = M;
    tC.shape[1] = N;
    tC.data = (void *)C_ptr;

    og_add_tensor(graph, &tA, &tid_A);
    og_add_tensor(graph, &tB, &tid_B);
    og_add_tensor(graph, &tC, &tid_C);

    // GEMM node
    og_node_t gemm = {};
    gemm.type = OG_OP_GEMM;
    gemm.input_ids[0] = tid_A;
    gemm.input_ids[1] = tid_B;
    gemm.num_inputs = 2;
    gemm.output_ids[0] = tid_C;
    gemm.num_outputs = 1;
    gemm.attributes[0] = alpha;
    gemm.attributes[1] = beta;
    gemm.num_attributes = 2;
    gemm.can_fuse_forward = 1;
    gemm.can_fuse_backward = 1;

    uint64_t nid_gemm;
    og_add_node(graph, &gemm, &nid_gemm);

    if (og_finalize_graph(graph) != OG_OK) {
      og_destroy_graph(graph);
      agee_destroy_session(session);
      return NPL_ERR_INTERNAL;
    }

    agee_graph_plan_t plan = nullptr;
    if (agee_create_plan_from_graph(session, graph, &plan) != AGEE_OK) {
      og_destroy_graph(graph);
      agee_destroy_session(session);
      return NPL_ERR_INTERNAL;
    }

    // Execute per-batch without changing batching semantics
    ParallelFor(0, batch_size, [&](size_t b) {
      void *inputs[2] = {
          (void *)(A_ptr + b * stride_A),
          (void *)(B_ptr + b * stride_B),
      };
      void *outputs[1] = {
          (void *)(C_ptr + b * stride_C),
      };
      agee_execute_plan_with_tensors(session, plan, inputs, 2, outputs, 1,
                                     nullptr);
    });

    if (stats) {
      double flops = 2.0 * batch_size * M * N * K;
      stats->gflops = flops / 1e9;
      stats->operations_fused = 1;
      stats->memory_saved_bytes = 0;
      stats->backend_used = "AGEE-Batched";
      stats->was_fused = 1;
    }

    agee_destroy_plan(plan);
    og_destroy_graph(graph);
    agee_destroy_session(session);

    return NPL_OK;
  }

  return NPL_OK;
}


int ExecuteElementwiseAdd(const npl_tensor_t *A, const npl_tensor_t *B,
                            npl_tensor_t *C, npl_perf_stats_t *stats) {

  size_t num_elements = GetTensorElementCount(C);
  const float *a_data = static_cast<const float *>(A->data);
  const float *b_data = static_cast<const float *>(B->data);
  float *c_data = static_cast<float *>(C->data);

  // Prefetch input/output tensors
  ffm_prefetch_block_read_T0(A->data, num_elements * sizeof(float));
  ffm_prefetch_block_read_T0(B->data, num_elements * sizeof(float));
  ffm_prefetch_block_write_T0(C->data, num_elements * sizeof(float));

  // TODO: Implement proper broadcasting support
  // For now, only handle same-shape contiguous tensors
  if (A->is_contiguous && B->is_contiguous && C->is_contiguous &&
      GetTensorElementCount(A) == num_elements &&
      GetTensorElementCount(B) == num_elements) {

    // Simple vectorized addition
    ParallelFor(0, num_elements, [&](size_t i) {
      c_data[i] = a_data[i] + b_data[i];
    });

      } else {
        // Broadcasting or non-contiguous tensors not yet supported
        return NPL_ERR_UNSUPPORTED;
      }

  if (stats) {
    stats->gflops = 0.0;
    stats->operations_fused = 0;
    stats->memory_saved_bytes = 0;
    stats->backend_used = "Vectorized";
    stats->was_fused = 0;
  }

  return NPL_OK;
}

int ExecuteElementwiseMul(const npl_tensor_t *A, const npl_tensor_t *B,
                            npl_tensor_t *C, npl_perf_stats_t *stats) {
  size_t num_elements = GetTensorElementCount(C);
  const float *a_data = static_cast<const float *>(A->data);
  const float *b_data = static_cast<const float *>(B->data);
  float *c_data = static_cast<float *>(C->data);

  size_t a_elements = GetTensorElementCount(A);
  size_t b_elements = GetTensorElementCount(B);

  // Prefetch input/output tensors
  ffm_prefetch_block_read_T0(A->data, a_elements * sizeof(float));
  ffm_prefetch_block_read_T0(B->data, b_elements * sizeof(float));
  ffm_prefetch_block_write_T0(C->data, num_elements * sizeof(float));

  // Case 1: Same-shape contiguous tensors - Use KFE elementwise_mul
  if (A->is_contiguous && B->is_contiguous && C->is_contiguous &&
      a_elements == num_elements && b_elements == num_elements &&
      A->ndim == B->ndim && A->ndim == C->ndim) {

    // Check if all shapes match
    bool same_shape = true;
    for (size_t i = 0; i < A->ndim; ++i) {
      if (A->shape[i] != B->shape[i] || A->shape[i] != C->shape[i]) {
        same_shape = false;
        break;
      }
    }

    if (same_shape) {
      size_t simd_width = kfe_eve_is_available() ? kfe_eve_simd_width() : 1;

      ParallelFor(0, num_elements / simd_width, [&](size_t block) {
        size_t start = block * simd_width;
        for (size_t i = 0; i < simd_width && start + i < num_elements; ++i) {
          c_data[start + i] = a_data[start + i] * b_data[start + i];
        }
      });

      // Scalar tail
      size_t vec_end = (num_elements / simd_width) * simd_width;
      for (size_t i = vec_end; i < num_elements; ++i) {
        c_data[i] = a_data[i] * b_data[i];
      }

      if (stats) {
        stats->backend_used = "ElementwiseMul+EVE";
        stats->was_fused = 0;
      }

      return NPL_OK;
    }
  }

  // Case 2: Broadcasting support - Scalar broadcast
  if (b_elements == 1) {
    // B is scalar - broadcast multiply
    float scalar = b_data[0];
    size_t simd_width = kfe_eve_is_available() ? kfe_eve_simd_width() : 1;

    ParallelFor(0, num_elements / simd_width, [&](size_t block) {
      size_t start = block * simd_width;
      for (size_t i = 0; i < simd_width && start + i < num_elements; ++i) {
        c_data[start + i] = a_data[start + i] * scalar;
      }
    });

    // Scalar tail
    size_t vec_end = (num_elements / simd_width) * simd_width;
    for (size_t i = vec_end; i < num_elements; ++i) {
      c_data[i] = a_data[i] * scalar;
    }

    if (stats) {
      stats->backend_used = "Broadcast-Scalar+EVE";
      stats->was_fused = 0;
    }

    return NPL_OK;
  }

  if (a_elements == 1) {
    // A is scalar - broadcast multiply
    float scalar = a_data[0];
    size_t simd_width = kfe_eve_is_available() ? kfe_eve_simd_width() : 1;

    ParallelFor(0, num_elements / simd_width, [&](size_t block) {
      size_t start = block * simd_width;
      for (size_t i = 0; i < simd_width && start + i < num_elements; ++i) {
        c_data[start + i] = scalar * b_data[start + i];
      }
    });

    size_t vec_end = (num_elements / simd_width) * simd_width;
    for (size_t i = vec_end; i < num_elements; ++i) {
      c_data[i] = scalar * b_data[i];
    }

    if (stats) {
      stats->backend_used = "Broadcast-Scalar+EVE";
      stats->was_fused = 0;
    }

    return NPL_OK;
  }

  // Case 3: Channel-wise broadcasting (e.g., [N,C,H,W] * [C,1,1] or [1,C,1,1])
  if (A->ndim == 4 && B->ndim == 4 && C->ndim == 4 &&
      A->shape[0] == C->shape[0] && A->shape[1] == B->shape[1] &&
      B->shape[0] == 1 && B->shape[2] == 1 && B->shape[3] == 1) {

    size_t batch = A->shape[0];
    size_t channels = A->shape[1];
    size_t spatial = A->shape[2] * A->shape[3];

    ParallelFor(0, batch * channels, [&](size_t idx) {
      size_t n = idx / channels;
      size_t c = idx % channels;

      float scale = b_data[c];
      const float *a_ptr = a_data + (n * channels + c) * spatial;
      float *c_ptr = c_data + (n * channels + c) * spatial;

      size_t simd_width = kfe_eve_is_available() ? kfe_eve_simd_width() : 1;
      size_t vec_end = (spatial / simd_width) * simd_width;

      for (size_t i = 0; i < vec_end; i += simd_width) {
        for (size_t j = 0; j < simd_width; ++j) {
          c_ptr[i + j] = a_ptr[i + j] * scale;
        }
      }

      for (size_t i = vec_end; i < spatial; ++i) {
        c_ptr[i] = a_ptr[i] * scale;
      }
    });

    if (stats) {
      stats->backend_used = "Broadcast-Channel+EVE";
      stats->was_fused = 0;
    }

    return NPL_OK;
  }

  // Case 4: Non-contiguous tensors - Generic strided multiplication
  if (!A->is_contiguous || !B->is_contiguous || !C->is_contiguous) {
    // Compute strides and handle non-contiguous access
    size_t indices[8] = {0};

    for (size_t i = 0; i < num_elements; ++i) {
      // Compute multi-dimensional indices
      size_t temp = i;
      for (int d = C->ndim - 1; d >= 0; --d) {
        indices[d] = temp % C->shape[d];
        temp /= C->shape[d];
      }

      // Compute linear offsets using strides
      size_t a_offset = 0, b_offset = 0, c_offset = 0;
      for (size_t d = 0; d < C->ndim; ++d) {
        size_t a_idx = (d < A->ndim) ? std::min(indices[d], A->shape[d] - 1) : 0;
        size_t b_idx = (d < B->ndim) ? std::min(indices[d], B->shape[d] - 1) : 0;

        a_offset += a_idx * A->strides[d];
        b_offset += b_idx * B->strides[d];
        c_offset += indices[d] * C->strides[d];
      }

      c_data[c_offset] = a_data[a_offset] * b_data[b_offset];
    }

    if (stats) {
      stats->backend_used = "Strided-Generic";
      stats->was_fused = 0;
    }

    return NPL_OK;
  }

  // Case 5: Unsupported broadcasting pattern
  return NPL_ERR_UNSUPPORTED;
}

/* ========================================================================== */
/* Utilities                                                                   */
/* ========================================================================== */

size_t GetTensorElementCount(const npl_tensor_t *tensor) {
  size_t count = 1;
  for (size_t i = 0; i < tensor->ndim; ++i) {
    count *= tensor->shape[i];
  }
  return count;
}

size_t GetElementSize(npl_dtype_t dtype) {
  switch (dtype) {
    case NPL_DTYPE_FP32: return 4;
    case NPL_DTYPE_FP16: return 2;
    case NPL_DTYPE_BF16: return 2;
    case NPL_DTYPE_INT8: return 1;
    case NPL_DTYPE_INT32: return 4;
    default: return 0;
  }
}

bool IsTensorContiguous(const npl_tensor_t *tensor) {
  return tensor->is_contiguous != 0;
}

bool CheckBroadcastable(const npl_tensor_t *A, const npl_tensor_t *B) {
  // Simplified check - assumes same shape or compatible broadcasting
  if (A->ndim != B->ndim) {
    return false;
  }

  for (size_t i = 0; i < A->ndim; ++i) {
    if (A->shape[i] != B->shape[i] && A->shape[i] != 1 && B->shape[i] != 1) {
      return false;
    }
  }

  return true;
}

void *AllocateAligned(size_t size, size_t alignment) {
  return ffm_aligned_alloc(alignment, size);
}

void FreeAligned(void *ptr) {
  ffm_free(ptr);
}

void *GetWorkspacePtr(size_t required_size) {
  std::lock_guard<std::mutex> lock(g_npl_state.workspace_mutex);

  if (required_size > g_npl_state.workspace_size) {
    return nullptr; // Not enough workspace
  }

  return g_npl_state.workspace;
}

void ReleaseWorkspace() {
  // Workspace is persistent - no explicit release needed
}

void StartTimer(std::chrono::high_resolution_clock::time_point *start) {
  *start = std::chrono::high_resolution_clock::now();
}

double GetElapsedMs(const std::chrono::high_resolution_clock::time_point &start) {
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return duration.count() / 1000.0;
}

double ComputeGFLOPS(double flops, double time_ms) {
  if (time_ms <= 0.0) return 0.0;
  return (flops / 1e9) / (time_ms / 1000.0);
}

int ValidateTensor(const npl_tensor_t *tensor, const char *name) {
  if (!tensor) {
    fprintf(stderr, "[NPL] Error: %s tensor is NULL\n", name);
    return NPL_ERR_INVALID_ARG;
  }

  if (!tensor->data) {
    fprintf(stderr, "[NPL] Error: %s tensor data is NULL\n", name);
    return NPL_ERR_INVALID_ARG;
  }

  if (tensor->ndim == 0 || tensor->ndim > NPL_MAX_DIMS) {
    fprintf(stderr, "[NPL] Error: %s tensor has invalid ndim: %zu\n",
            name, tensor->ndim);
    return NPL_ERR_INVALID_ARG;
  }

  for (size_t i = 0; i < tensor->ndim; ++i) {
    if (tensor->shape[i] == 0) {
      fprintf(stderr, "[NPL] Error: %s tensor has zero dimension at axis %zu\n",
              name, i);
      return NPL_ERR_INVALID_ARG;
    }
  }

  return NPL_OK;
}

void PrintTensorInfo(const npl_tensor_t *tensor, const char *name) {
  printf("[NPL] Tensor '%s':\n", name);
  printf("  Shape: [");
  for (size_t i = 0; i < tensor->ndim; ++i) {
    printf("%zu%s", tensor->shape[i], i < tensor->ndim - 1 ? ", " : "");
  }
  printf("]\n");
  printf("  Data type: %d\n", tensor->dtype);
  printf("  Layout: %d\n", tensor->layout);
  printf("  Size: %zu bytes\n", tensor->size_bytes);
  printf("  Contiguous: %s\n", tensor->is_contiguous ? "Yes" : "No");
}

void LogOperation(const char *op_name, const npl_perf_stats_t *stats) {
  if (!g_npl_state.config.verbose || !stats) {
    return;
  }

  printf("[NPL] %s: %.3f ms, %.2f GFLOPS%s\n",
         op_name, stats->elapsed_ms, stats->gflops,
         stats->was_fused ? " (FUSED)" : "");
}

} // namespace NPL_internal