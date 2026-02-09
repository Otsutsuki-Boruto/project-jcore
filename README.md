# **Project JCore**

**JCore** is a modular high-performance neural compute engine providing optimized primitives for matrix/tensor algebra, convolution, normalization, and fused execution.

It is designed as a layered system built on top of BLAS backends (OpenBLAS/BLIS) and integrates:

* cache-aware tiling
* kernel dispatch
* polyhedral optimization
* JIT kernel generation
* operator fusion
* graph execution

The original objective was to expose these capabilities to Java through the FFM API. The current repository delivers the complete native engine and serves as a standalone high-performance backend and research platform.

---

## **Architecture (high level)**

```
Neural Primitives Layer (NPL)
        ↓
Polyhedral + JIT kernels
        ↓
Graph Execution Engine (Operator Graph + Kernel Fusion Engine)
        ↓
Microkernel Interface Layer
```

---

## **Features**

### **Matrix / Tensor Operations**

* `npl_matmul`
* `npl_add`, `npl_mul`
* broadcasting
* element-wise ops

### **Convolutions**

* `npl_conv2d`
* `npl_depthwise_conv2d`
* `npl_conv2d_bn_activation` (fused)

### **Pooling**

* `npl_pooling`
* `npl_global_avg_pool`

### **Normalization**

* `npl_batch_norm`
* `npl_layer_norm`

### **Activations**

* `npl_relu`
* `npl_leaky_relu`
* `npl_softmax`
* generic `npl_activation`

### **Tensor Runtime**

* `npl_create_tensor`
* `npl_allocate_tensor`
* `npl_free_tensor`
* `npl_reshape_tensor`

---

## **Performance Notes**

**JCore** builds on top of BLAS kernels and applies:

* dispatch selection
* tiling
* fusion
* JIT specialization

For medium/large workloads this typically provides **higher effective throughput than direct BLAS calls**, particularly when multiple operations can be fused.

It is **not intended to replace vendor libraries such as Intel MKL**, but to provide a modular and extensible open-source alternative for experimentation and research.

---

## **Requirements**

### **System libraries:**

* OpenBLAS
* BLIS
* TBB
* PAPI
* NUMA
* SLEEF
* memkind
* jemalloc
* pthread, dl, m
* EVE / VectorClass / Highway
* LLVM 20.x (Polly/MLIR optional)

Minimum ISA: **AVX2**

### **Project Libraries:**

Project dependent libraries which are the submodules of project and are a necessity when working it by **native integration** of **FFM API**. You can recompile
the static and shared libraries for the project (The libs compilation commands are available in compile directory).

Alternatively libraries are made available including **LLVM 20.1.0** static and shared libraries in project releases.
---

## **Build**

```bash
mkdir build
cd build
cmake ..
make -j
```

---

## **Usage**

Include:

```
advanced/neural_layer/include/neural_primitives.h
```

Initialize:

```cpp
npl_init(nullptr);
```

Example usage:

```
project_test/neural_layer_test.cpp
```

---

## Status

This repository represents a **stable, feature-complete native backend**.
Future work may include language bindings (Java FFM) and additional ISA targets.

---