# **Compile.md**

## **Kernel Fusion Engine**

```bash
# compile
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/mkernel_interface/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_allocator/include \
  -Ibase/mem_prefetch/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  advanced/kFusion_engine/src/*.cpp \
  -Ladvanced/lib/static \
  -Lbase/mem_allocator \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -lmkernel_interface -lvector_math_engine -ldispatch_kernel -ltuning_rCache \
  -lkernel_autoTuner -lglobal_thread -lnuma_manager -lpool_manager \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -lconfig_env \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lthread_scheduler \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o advanced/kFusion_engine/kernel_fusion
```

```bash
# static libraries
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/mkernel_interface/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_allocator/include \
  -Ibase/mem_prefetch/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -c advanced/kFusion_engine/src/kernel_fusion_engine_core.cpp \
     advanced/kFusion_engine/src/kernel_fusion_helpers.cpp \
     advanced/kFusion_engine/src/kernel_fusion_ops.cpp \
     advanced/kFusion_engine/src/kernel_fusion_utils.cpp \
     advanced/kFusion_engine/src/kernel_fusion_eve.cpp
ar rcs libkernel_fusion.a \
  kernel_fusion_engine_core.o \
  kernel_fusion_helpers.o kernel_fusion_ops.o \
  kernel_fusion_utils.o kernel_fusion_eve.o

```

```bash
# shared libraries
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE -shared -o libkernel_fusion.so \
  kernel_fusion_engine_core.o \
  kernel_fusion_helpers.o \
  kernel_fusion_ops.o \
  kernel_fusion_utils.o \
  kernel_fusion_eve.o \
  -Ladvanced/lib/shared \
  -Lcomposite/lib/shared \
  -Lbase/mem_allocator \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -lmkernel_interface -lvector_math_engine -ldispatch_kernel -ltuning_rCache \
  -lkernel_autoTuner -lglobal_thread -lnuma_manager -lpool_manager \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -lconfig_env \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lthread_scheduler \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc \
  -lpthread -ldl -lm

```

## **Operator Graph/Fusion Runtime**

```bash
# Compile
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/vmath_engine/include \
  -Icomposite/mkernel_interface/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/config_env/include \
  -Ibase/thread_schedule/include \
  -Ibase/cache_block/include \
  -Ibase/mem_prefetch/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  advanced/operator_graph/src/*.cpp \
  -Ladvanced/lib/static \
  -Lbase/mem_allocator \
  -lkernel_fusion -lmkernel_interface -lvector_math_engine -lglobal_thread -ldispatch_kernel \
  -ltuning_rCache -lkernel_autoTuner -lmicro_timer -lnuma_manager -lpool_manager -lthread_scheduler \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lconfig_env \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o advanced/operator_graph/operator_graph
```

```bash
# static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/vmath_engine/include \
  -Icomposite/mkernel_interface/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/config_env/include \
  -Ibase/thread_schedule/include \
  -Ibase/cache_block/include \
  -Ibase/mem_prefetch/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -c advanced/operator_graph/src/operator_graph_analysis.cpp \
     advanced/operator_graph/src/operator_graph_construction.cpp \
     advanced/operator_graph/src/operator_graph_core.cpp \
     advanced/operator_graph/src/operator_graph_execution.cpp \
     advanced/operator_graph/src/operator_graph_utils.cpp \

  ar rcs advanced/operator_graph/liboperator_graph.a \
  operator_graph_analysis.o operator_graph_construction.o operator_graph_core.o \
  operator_graph_execution.o operator_graph_utils.o
```

```bash
# shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE -shared -o advanced/operator_graph/liboperator_graph.so \
  operator_graph_analysis.o \
  operator_graph_construction.o \
  operator_graph_core.o \
  operator_graph_execution.o \
  operator_graph_utils.o \
  -Ladvanced/lib/shared \
  -lkernel_fusion -lmkernel_interface -lvector_math_engine -lglobal_thread -ldispatch_kernel \
  -ltuning_rCache -lkernel_autoTuner -lmicro_timer -lnuma_manager -lpool_manager -lthread_scheduler \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lconfig_env \
  -lopenblas -lblis -lxsmm \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm
```

## **JIT Kernel Generator**

```bash
# compile
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -I/usr/lib/llvm-20/include \
  -I/usr/lib/llvm-20/include/llvm-c \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/vmath_engine/include \
  -Icomposite/mkernel_interface/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/config_env/include \
  -Ibase/thread_schedule/include \
  -Ibase/cache_block/include \
  -Ibase/mem_prefetch/include \
  -Ibase/isa_aware/include \
  -Ibase/mem_allocator/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  advanced/jit_kernel/src/*.cpp \
  -Ladvanced/lib/static \
  -lkernel_fusion -lmkernel_interface -lvector_math_engine -ldispatch_kernel \
  -ltuning_rCache -lkernel_autoTuner -lglobal_thread -lnuma_manager -lpool_manager \
  -lmicro_timer -lthread_scheduler -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -Wl,--no-as-needed -lzstd -lz -ltinfo -lrt -Wl,--as-needed \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc \
  -lpthread -ldl -lm \
  -o advanced/jit_kernel/jit_kernel
```

```bash
# static lib
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -I/usr/lib/llvm-20/include \
  -I/usr/lib/llvm-20/include/llvm-c \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/vmath_engine/include \
  -Icomposite/mkernel_interface/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/config_env/include \
  -Ibase/thread_schedule/include \
  -Ibase/cache_block/include \
  -Ibase/mem_prefetch/include \
  -Ibase/isa_aware/include \
  -Ibase/mem_allocator/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -c advanced/jit_kernel/src/jit_core.cpp \
     advanced/jit_kernel/src/llvm_codegen.cpp \
     advanced/jit_kernel/src/vectorization_backends.cpp \
     advanced/jit_kernel/src/jit_utilities.cpp \
     advanced/jit_kernel/src/jit_wrapper.cpp \

  ar rcs advanced/jit_kernel/libjit_kernel_generator.a \
  jit_core.o llvm_codegen.o vectorization_backends.o \
  jit_utilities.o jit_wrapper.o
```

```bash
# shared lib
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE -shared -o advanced/jit_kernel/libjit_kernel_generator.so \
   jit_core.o llvm_codegen.o vectorization_backends.o \
  jit_utilities.o jit_wrapper.o \
  -Ladvanced/lib/shared \
  -lkernel_fusion -lmkernel_interface -lvector_math_engine -ldispatch_kernel \
  -ltuning_rCache -lkernel_autoTuner -lpool_manager -lglobal_thread -lnuma_manager \
  -lmicro_timer -lthread_scheduler -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -Wl,--no-as-needed -lzstd -lz -ltinfo -lrt -Wl,--as-needed \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm
```

## **Adaptive Graph Execution Engine**

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/graph_execution/include \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/mkernel_interface/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/perf_profiler/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/config_env/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_prefetch/include \
  -Ibase/mem_allocator/include \
  -Ibase/cache_block/include \
  -Ibase/hp_controller/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  advanced/graph_execution/src/*.cpp \
  -Ladvanced/lib/static \
  -loperator_graph -lkernel_fusion -lmkernel_interface -ltuning_rCache -lvector_math_engine -ldispatch_kernel \
  -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager -lpool_manager \
  -lmicro_timer -lthread_scheduler -lffm -lffm_cache_block -lffm_prefetch -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -lopenblas -lblis \
  -ltbb -lpapi -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o advanced/graph_execution/graph_execution_engine
```

```bash
# static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/graph_execution/include \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/mkernel_interface/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/perf_profiler/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/config_env/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_prefetch/include \
  -Ibase/mem_allocator/include \
  -Ibase/cache_block/include \
  -Ibase/hp_controller/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -c advanced/graph_execution/src/agee_core.cpp \
     advanced/graph_execution/src/agee_executor.cpp \
     advanced/graph_execution/src/agee_graph_analysis.cpp \
     advanced/graph_execution/src/agee_memory_manager.cpp \
     advanced/graph_execution/src/agee_scheduler.cpp \

  ar rcs advanced/graph_execution/libgraph_execution_engine.a \
  agee_core.o agee_executor.o agee_graph_analysis.o \
  agee_memory_manager.o agee_scheduler.o
```

```bash
# shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE -shared -o advanced/graph_execution/libgraph_execution_engine.so \
  agee_core.o agee_executor.o agee_graph_analysis.o \
  agee_memory_manager.o agee_scheduler.o \
  -Ladvanced/lib/shared \
  -loperator_graph -lkernel_fusion -lmkernel_interface -ltuning_rCache -lvector_math_engine -ldispatch_kernel \
  -ltuning_rCache -lkernel_autoTuner -lpool_manager -lmicro_timer -lthread_scheduler \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -Wl,--no-as-needed -lzstd -lz -ltinfo -lrt -Wl,--as-needed \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm
```

##  Neural Primitives Layer

```bash
# Compile NPL
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -I/usr/lib/llvm-20/include \
  -Iadvanced/neural_layer/include \
  -Iadvanced/polyhedral_optimization/include \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/graph_execution/include \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/mkernel_interface/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_prefetch/include \
  -Ibase/mem_allocator/include \
  -Ibase/cache_block/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -Ibase/config_env/include \
  advanced/neural_layer/src/*.cpp \
  -Ladvanced/lib/static \
  -lpolyhedral_optimization -ljit_kernel_generator -lgraph_execution_engine -loperator_graph -lkernel_fusion \
  -lmkernel_interface -ltuning_rCache -lvector_math_engine -ldispatch_kernel \
  -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager -lpool_manager \
  -lmicro_timer -lthread_scheduler -lffm -lffm_cache_block -lffm_prefetch -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -ltbb -lpapi -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o advanced/neural_layer/neural_layer
```

```bash
# static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -I/usr/lib/llvm-20/include \
  -Iadvanced/neural_layer/include \
  -Iadvanced/polyhedral_optimization/include \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/graph_execution/include \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/mkernel_interface/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_prefetch/include \
  -Ibase/mem_allocator/include \
  -Ibase/cache_block/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -Ibase/config_env/include \
  -c advanced/neural_layer/src/npl_activation_norm.cpp \
     advanced/neural_layer/src/npl_convolution.cpp \
     advanced/neural_layer/src/npl_core.cpp \
     advanced/neural_layer/src/npl_pooling.cpp \
     advanced/neural_layer/src/npl_tensor_ops.cpp \

  ar rcs advanced/neural_layer/libneural_layer.a \
  npl_activation_norm.o npl_convolution.o npl_core.o \
  npl_pooling.o npl_tensor_ops.o
```

```bash
# shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE -shared -o advanced/neural_layer/libneural_layer.so \
  npl_activation_norm.o npl_convolution.o npl_core.o \
  npl_pooling.o npl_tensor_ops.o \
  -Ladvanced/lib/shared \
  -lpolyhedral_optimization -ljit_kernel_generator -lgraph_execution_engine -loperator_graph -lkernel_fusion \
  -lmkernel_interface -ltuning_rCache -lvector_math_engine -ldispatch_kernel \
  -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager -lpool_manager \
  -lmicro_timer -lthread_scheduler -lffm -lffm_cache_block -lffm_prefetch -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -ltbb -lpapi -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm
```

## **Polyhedral Optimization Layer**

```bash
# Compile
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -I/usr/lib/llvm-20/include \
  -Iadvanced/polyhedral_optimization/include \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/mkernel_interface/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/config_env/include \
  -Ibase/thread_schedule/include \
  -Ibase/cache_block/include \
  -Ibase/mem_prefetch/include \
  -Ibase/isa_aware/include \
  -Ibase/mem_allocator/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  advanced/polyhedral_optimization/src/*.cpp \
  -Ladvanced/lib/static \
  -ljit_kernel_generator -lkernel_fusion -lmkernel_interface \
  -lvector_math_engine -ldispatch_kernel \
  -ltuning_rCache -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager \
  -lpool_manager -lmicro_timer -lthread_scheduler \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  -lopenblas -lblis \
  -lzstd -lz -ltinfo -lrt -Wl,--as-needed \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc \
  -lpthread -ldl -lm \
  -o advanced/polyhedral_optimization/polyhedral_optimize
```

```bash
# static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -I/usr/lib/llvm-20/include \
  -Iadvanced/polyhedral_optimization/include \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/mkernel_interface/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/config_env/include \
  -Ibase/thread_schedule/include \
  -Ibase/cache_block/include \
  -Ibase/mem_prefetch/include \
  -Ibase/isa_aware/include \
  -Ibase/mem_allocator/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -c advanced/polyhedral_optimization/src/polyhedral_core.cpp \
     advanced/polyhedral_optimization/src/polyhedral_vectorization.cpp \
     advanced/polyhedral_optimization/src/polyhedral_llvm_integration.cpp \
     advanced/polyhedral_optimization/src/polyhedral_tiling.cpp \

  ar rcs advanced/polyhedral_optimization/libpolyhedral_optimization.a \
  polyhedral_core.o polyhedral_vectorization.o \
  polyhedral_llvm_integration.o polyhedral_tiling.o

```

```bash
# shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE -shared -o advanced/polyhedral_optimization/libpolyhedral_optimization.so \
  polyhedral_core.o polyhedral_vectorization.o polyhedral_llvm_integration.o polyhedral_tiling.o\
  -Ladvanced/lib/shared \
  -ljit_kernel_generator -lkernel_fusion -lmkernel_interface \
  -lvector_math_engine -ldispatch_kernel \
  -ltuning_rCache -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager \
  -lpool_manager -lmicro_timer -lthread_scheduler \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  -lopenblas -lblis \
  -lzstd -lz -ltinfo -lrt -Wl,--as-needed \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc \
  -lpthread -ldl -lm

```

## **Self Optimizing Runtime**

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/selfOptimizing_runtime/include \
  -Iadvanced/neural_layer/include \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/graph_execution/include \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/mkernel_interface/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/perf_profiler/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_prefetch/include \
  -Ibase/mem_allocator/include \
  -Ibase/cache_block/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -Ibase/config_env/include \
  advanced/selfOptimizing_runtime/src/*.cpp \
  -Ladvanced/lib/static \
  -ljcore_neural_layer -ljit_kernel_generator -lgraph_execution_engine -loperator_graph -lkernel_fusion \
  -lmkernel_interface -ltuning_rCache -lvector_math_engine -ldispatch_kernel \
  -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager -lpool_manager \
  -lmicro_timer -lthread_scheduler -lffm -lffm_cache_block -lffm_prefetch -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -ltbb -lpapi -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o advanced/selfOptimizing_runtime/self_optimize
```

```bash
# static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/selfOptimizing_runtime/include \
  -Iadvanced/neural_layer/include \
  -Iadvanced/polyhedral_optimization/include \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/graph_execution/include \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/mkernel_interface/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/perf_profiler/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_prefetch/include \
  -Ibase/mem_allocator/include \
  -Ibase/cache_block/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -Ibase/config_env/include \
  -c advanced/selfOptimizing_runtime/src/sor_adaptation.cpp \
     advanced/selfOptimizing_runtime/src/sor_benchmarking.cpp \
     advanced/selfOptimizing_runtime/src/sor_core.cpp \
     advanced/selfOptimizing_runtime/src/sor_learning.cpp \
     advanced/selfOptimizing_runtime/src/sor_tuning_cache.cpp \

  ar rcs advanced/selfOptimizing_runtime/libself_optimizing_runtime.a \
  sor_adaptation.o sor_benchmarking.o sor_core.o sor_learning.o sor_tuning_cache.o
```

```bash
# shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE -shared -o advanced/selfOptimizing_runtime/libself_optimizing_runtime.so \
  sor_adaptation.o sor_benchmarking.o sor_core.o sor_learning.o sor_tuning_cache.o \
  -Ladvanced/lib/shared \
  -ljcore_neural_layer -lpolyhedral_optimization -ljit_kernel_generator -lgraph_execution_engine -loperator_graph -lkernel_fusion \
  -lmkernel_interface -ltuning_rCache -lvector_math_engine -ldispatch_kernel \
  -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager -lpool_manager \
  -lmicro_timer -lthread_scheduler -lffm -lffm_cache_block -lffm_prefetch -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -ltbb -lpapi -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm 
```
