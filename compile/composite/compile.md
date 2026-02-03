# **Compilation and Shared libraries**

## **Adaptive Kernel AutoTuner**

```bash
# compile

# please ensure you have test_tuner.cpp file to create executable:
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/kernel_autoTuner/src/adaptive_tuner.cpp \
  composite/kernel_autoTuner/src/kernels_blis.cpp \
  composite/kernel_autoTuner/src/kernels_openblas.cpp \
  composite/kernel_autoTuner/src/test_tuner.cpp \
  -Lcomposite/lib/static \
  -lglobal_thread -lnuma_manager -lpool_manager \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -lffm -lffm_huge -lffm_prefetch -lffm_cache_block \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/kernel_autoTuner/kernel_tuner

```

```bash
# create static library:

# Compile object files
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -c \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/kernel_autoTuner/src/adaptive_tuner.cpp \
  composite/kernel_autoTuner/src/kernels_blis.cpp \
  composite/kernel_autoTuner/src/kernels_openblas.cpp \

# Move object files to kernel_autoTuner
mv *.o composite/kernel_autoTuner/

# Create static library
ar rcs composite/kernel_autoTuner/libkernel_autoTuner.a \
  composite/kernel_autoTuner/adaptive_tuner.o \
  composite/kernel_autoTuner/kernels_blis.o \
  composite/kernel_autoTuner/kernels_openblas.o
```

```bash
# create shared library:
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -shared -o composite/kernel_autoTuner/libkernel_autoTuner.so \
  composite/kernel_autoTuner/adaptive_tuner.o \
  composite/kernel_autoTuner/kernels_blis.o \
  composite/kernel_autoTuner/kernels_openblas.o \
  -Lcomposite/lib/shared \
  -lglobal_thread -lnuma_manager -lpool_manager \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -lffm -lffm_huge -lffm_prefetch -lffm_cache_block \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lmemkind -ljemalloc -lpthread -ldl -lm
``
```

## **Numa Aware Memory Manager**

```bash
# compile

g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/numa_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/hp_controller/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  composite/numa_manager/src/numa_memory_manager.cpp \
  composite/numa_manager/src/test_numa_memory_manager.cpp \
  -Lbase/mem_allocator \
  -Lcomposite/lib/static \
  -Lcomposite/lib/shared \
  -lffm -lffm_huge -lffm_prefetch -lffm_cache_block \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -lnuma -lmemkind -ljemalloc \
  -ltbb -lpthread -ldl -lm \
  -Wl,-rpath=$(pwd)/composite/lib/shared \
  -o composite/numa_manager/numa_manager_testlpthread

```

```bash
# create static library

g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/hp_controller/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -c composite/numa_manager/src/numa_memory_manager.cpp \
  -o composite/numa_manager/numa_memory_manager.o

  ar rcs composite/numa_manager/libnuma_manager.a \
  composite/numa_manager/numa_memory_manager.o

```

```bash
# create shared library

g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -shared \
  composite/numa_manager/numa_memory_manager.o \
  -Lbase/mem_allocator -lffm -lffm_huge -lffm_prefetch -lffm_cache_block \
  -Lcomposite/lib/shared -lpool_manager -lconfig_env -lmicro_timer -lthread_scheduler \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lnuma -lmemkind -ljemalloc -ltbb -lpthread -ldl -lm \
  -o composite/numa_manager/libnuma_manager.so \
  -Wl,-rpath=$(pwd)/composite/lib/shared
```

## **Memory Pool Manager**

```bash
# compile
# please ensure you have pool_manager_test.c file to create executable:

g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/pool_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/hp_controller/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  composite/pool_manager/src/pool_manager.c \
  composite/pool_manager/src/pool_manager_test.c \
  -Lbase/mem_allocator \
  -Lcomposite/lib/static \
  -Lcomposite/lib/shared \
  -lffm -lffm_huge -ljcore_hw_introspect -lconfig_env \
  -lffm_prefetch -lffm_cache_block -ljcore_isa_aware \
  -lcpu_detect -lmicro_timer -lthread_scheduler \
  -lnuma -lmemkind -ljemalloc \
  -ltbb -lpthread -ldl -lm \
  -Wl,-rpath=$(pwd)/composite/lib/shared \
  -o composite/pool_manager/pool_manager_test
```

```bash
# create static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/pool_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/hp_controller/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -c composite/pool_manager/src/*.cpp \
  -o composite/pool_manager/pool_manager.o

  ar rcs composite/pool_manager/libpool_manager.a \
  composite/pool_manager/pool_manager.o
```

```bash
# create shared library
# Link into a shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -shared -o composite/pool_manager/libpool_manager.so \
  composite/pool_manager/pool_manager.o \
  -Lcomposite/lib/static \
  -Lbase/mem_allocator \
  -lffm -lffm_huge -ljcore_hw_introspect -lconfig_env \
  -lffm_prefetch -lffm_cache_block -ljcore_isa_aware \
  -lcpu_detect -lmicro_timer -lthread_scheduler \
  -lnuma -lmemkind -ljemalloc \
  -ltbb -lpthread -ldl -lm
```

## **Kernel Dispatcher/Runtime Selector**

```bash
# compile

# do not use -march=native flag which can disable some hand-tuned library-level vectorization
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/dispatch_kernel/src/*.cpp \
  -Lcomposite/lib/static \
  -lglobal_thread -lnuma_manager -lpool_manager -lkernel_autoTuner\
   -lmicro_timer -lthread_scheduler \
  -lffm -lffm_huge -lffm_prefetch -lffm_cache_block \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lconfig_env \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/dispatch_kernel/kernel_dispatch
```

```bash
# create static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
      -Icomposite/kernel_autoTuner/include \
      -Icomposite/dispatch_kernel/include \
      -Icomposite/global_thread/include \
      -Icomposite/numa_manager/include \
      -Icomposite/pool_manager/include \
      -Icomposite/include \
      -Ibase/benchmark/include \
      -Ibase/thread_schedule/include \
      -Ibase/cpu_detect/include \
      -Ibase/mem_allocator/include \
      -Ibase/isa_aware/include \
      -Ibase/cache_block/include \
      -Ibase/config_env/include \
      -Ibase/hw_introspect/include \
      -Ibase/mem_prefetch/include \
  -c composite/dispatch_kernel/src/jcore_dispatch.cpp \
  composite/dispatch_kernel/src/k_adaptive_tuner.cpp \
  composite/dispatch_kernel/src/k_cpu_features.cpp \
  composite/dispatch_kernel/src/k_isa_dispatch.cpp \
  composite/dispatch_kernel/src/k_kernel_dispatch.cpp \
  composite/dispatch_kernel/src/k_mock_kernels.cpp

# Move object files to kernel_autoTuner
mv *.o composite/dispatch_kernel/

# Create static library
ar rcs composite/dispatch_kernel/libdispatch_kernel.a \
  composite/dispatch_kernel/jcore_dispatch.o \
  composite/dispatch_kernel/k_adaptive_tuner.o \
  composite/dispatch_kernel/k_cpu_features.o \
  composite/dispatch_kernel/k_isa_dispatch.o \
  composite/dispatch_kernel/k_kernel_dispatch.o \
  composite/dispatch_kernel/k_mock_kernels.o

```

```bash
# create shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -shared -o composite/dispatch_kernel/libdispatch_kernel.so \
  composite/dispatch_kernel/jcore_dispatch.o \
  composite/dispatch_kernel/k_adaptive_tuner.o \
  composite/dispatch_kernel/k_cpu_features.o \
  composite/dispatch_kernel/k_isa_dispatch.o \
  composite/dispatch_kernel/k_kernel_dispatch.o \
  composite/dispatch_kernel/k_mock_kernels.o \
  -Lcomposite/lib/shared \
  -Lbase/mem_allocator \
  -lpool_manager -lglobal_thread -lnuma_manager -lkernel_autoTuner \
  -lffm -lffm_huge -ljcore_hw_introspect -lconfig_env \
  -lffm_prefetch -lffm_cache_block -ljcore_isa_aware \
  -lcpu_detect -lmicro_timer -lthread_scheduler \
  -lopenblas -lblis \
  -lnuma -lhwloc -lmemkind -ljemalloc -ltbb -lpthread -ldl -lm

```

## **Vector Math Engine**

```bash
# compile
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/include/ \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/vmath_engine/src/*.cpp \
  -Lcomposite/lib/static \
  -Lbase/mem_allocator \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -ldispatch_kernel \
  -lkernel_autoTuner \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/vmath_engine/vmath_engine
```

```bash
# static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/include/ \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  -c composite/vmath_engine/src/vmath_engine.cpp \
     composite/vmath_engine/src/vmath_fallback.cpp \
     composite/vmath_engine/src/vmath_fused.cpp \
     composite/vmath_engine/src/vmath_sleef_isolated.cpp \
     composite/vmath_engine/src/vmath_sleef_wrapper.cpp

# Step 2: Create static library
ar rcs composite/vmath_engine/libvector_math_engine.a *.o

```

```bash
# create shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/include/ \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/vmath_engine/src/vmath_engine.cpp \
  composite/vmath_engine/src/vmath_fallback.cpp \
  composite/vmath_engine/src/vmath_fused.cpp \
  composite/vmath_engine/src/vmath_sleef_isolated.cpp \
  composite/vmath_engine/src/vmath_sleef_wrapper.cpp \
  -Lcomposite/lib/shared \
  -Lbase/mem_allocator \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -ldispatch_kernel \
  -lkernel_autoTuner \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -shared -o composite/vmath_engine/libvector_math_engine.so
```

## **Global Thread Scheduler**

```bash
# compile
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include/ \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/global_thread/src/*.cpp \
  -Lcomposite/lib/static \
  -Lbase/mem_allocator \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -ldispatch_kernel \
  -lkernel_autoTuner \
  -lnuma_manager \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -lconfig_env \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lthread_scheduler \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/global_thread/global_thread
```

```bash
# create static libraries only if there is one file only.

g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include/ \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  -c composite/global_thread/src/*.cpp -o composite/global_thread/global_thread.o

ar rcs composite/global_thread/libglobal_thread.a composite/global_thread/*.o
```

```bash
# create shared libraries
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -shared \
  composite/global_thread/*.o \
  -Lcomposite/lib/shared \
  -Lbase/mem_allocator \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -ldispatch_kernel -lkernel_autoTuner -lnuma_manager -lpool_manager \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -lconfig_env \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lthread_scheduler \
  -lopenblas -lblis -lxsmm \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/global_thread/libglobal_thread.so
```

## **Performance Profiler / Telemetry**

```bash
# compile

```

```bash
# static libraries
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -c \
  -Icomposite/perf_profiler/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include/ \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/perf_profiler/src/profiler_core.cpp \
  composite/perf_profiler/src/profiler_region.cpp \
  composite/perf_profiler/src/profiler_papi.cpp \
  composite/perf_profiler/src/profiler_export.cpp \
  composite/perf_profiler/src/profiler_kernel_dispatch.cpp

ar rcs libperf_profiler.a \
  profiler_core.o profiler_region.o profiler_papi.o profiler_export.o profiler_kernel_dispatch.o
```

```bash
# shared libraries
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -shared -o libperf_profiler.so \
  profiler_core.o profiler_region.o profiler_papi.o \
  profiler_export.o profiler_kernel_dispatch.o \
  -Lcomposite/lib/shared \
  -Lbase/mem_allocator \
  -Lbase/isa_aware \
  -Lbase/cache_block \
  -Lbase/thread_schedule \
  -Lbase/benchmark \
  -Lbase/hw_introspect \
  -Lbase/cpu_detect \
  -Lbase/mem_prefetch \
  -Lbase/config_env \
  -Wl,--start-group \
  -ldispatch_kernel \
  -lkernel_autoTuner -ljcore_isa_aware -lglobal_thread -lnuma_manager -lpool_manager \
  -lthread_scheduler -lmicro_timer -ljcore_hw_introspect -lcpu_detect \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -lconfig_env \
  -Wl,--end-group \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpapi -lpthread -ldl -lm

```

## **Tuning Result and Cache System**

```bash
# compile

g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/include/ \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/tuning_rCache/src/*.cpp \
  -Lcomposite/lib/static \
  -Lbase/mem_allocator \
  -lglobal_thread -lnuma_manager -lkernel_autoTuner -lmicro_timer -lthread_scheduler \
  -ldispatch_kernel -lkernel_autoTuner -lnuma_manager \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -lconfig_env \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lthread_scheduler \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/tuning_rCache/tuned_cache
```

```bash
# static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  -c composite/tuning_rCache/src/cached_autotuner.cpp \
  composite/tuning_rCache/src/tuning_cache.cpp

# Create the static library
ar rcs libtuning_rCache.a cached_autotuner.o tuning_cache.o

```

```bash
# shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -shared -o libtuning_rCache.so *.o \
  -Lcomposite/lib/shared \
  -Lbase/mem_allocator \
  -lpool_manager -lglobal_thread -lnuma_manager -lkernel_autoTuner \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -ldispatch_kernel \
  -lkernel_autoTuner \
  -lnuma_manager \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc \
  -lpthread -ldl -lm

```

## **Microkernel Interface Layer**

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/mkernel_interface/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include/ \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/mkernel_interface/src/*.cpp \
  -Lcomposite/lib/static \
  -Lbase/mem_allocator \
  -lvector_math_engine -ldispatch_kernel -ltuning_rCache -lkernel_autoTuner -lpool_manager\
  -lthread_scheduler -lmicro_timer -lconfig_env \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc \
  -lpthread -ldl -lm \
  -o composite/mkernel_interface/microkernel
```

```bash
# static library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/mkernel_interface/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/pool_manager/include \
  -Icomposite/include/ \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  -c composite/mkernel_interface/src/mil_conv.cpp \
     composite/mkernel_interface/src/mil_core.cpp \
     composite/mkernel_interface/src/mil_gemm.cpp \
     composite/mkernel_interface/src/mil_gemv.cpp \
     composite/mkernel_interface/src/mil_vector.cpp \
     composite/mkernel_interface/src/mil_util.cpp

ar rcs libmkernel_interface.a mil_conv.o mil_core.o mil_gemm.o mil_gemv.o mil_vector.o mil_util.o

```

```bash
# shared library
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -shared -o libmkernel_interface.so \
  mil_conv.o mil_core.o mil_gemm.o mil_gemv.o mil_vector.o mil_util.o \
  -Lcomposite/lib/shared \
  -Lbase/mem_allocator \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -lpool_manager -lvector_math_engine -ldispatch_kernel -ltuning_rCache -lkernel_autoTuner \
  -lglobal_thread -lnuma_manager -lffm_cache_block -lffm -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc \
  -lpthread -ldl -lm

```
