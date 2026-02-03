# **Compilation Command:**

```bash
# compile : no avx512 flag for now!
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
  -lopenblas -lblis -lxsmm \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/global_thread/global_thread
```
