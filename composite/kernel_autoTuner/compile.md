# **compile**

```bash
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
