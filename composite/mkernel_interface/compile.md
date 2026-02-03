# **Compile**

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
