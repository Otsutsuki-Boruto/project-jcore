# Compile

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC  \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/tuning_rCache/include \
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
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -ldispatch_kernel \
  -lkernel_autoTuner \
  -lnuma_manager \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -lconfig_env \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lthread_scheduler \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/tuning_rCache/tuned_cache
```
