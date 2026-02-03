# **Compilation Command**

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC  \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/numa_manager/include \
  -Icomposite/global_thread/include \
  -Icomposite/perf_profiler/include \
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
  composite/perf_profiler/src/*.cpp \
  -Lcomposite/lib/static \
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
  -lkernel_autoTuner \
  -ljcore_isa_aware \
  -lglobal_thread \
  -lnuma_manager \
  -lthread_scheduler \
  -lmicro_timer \
  -ljcore_hw_introspect \
  -lcpu_detect \
  -lffm_cache_block \
  -lffm \
  -lffm_huge \
  -lffm_prefetch \
  -lconfig_env \
  -Wl,--end-group \
  -lopenblas -lblis -lxsmm \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpapi -lpthread -ldl -lm \
  -o composite/perf_profiler/perf_profiler

```
