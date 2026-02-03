# **Compile**

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/mkernel_interface/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/kernel_autoTuner/include \
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
  -lmkernel_interface -lvector_math_engine -ldispatch_kernel \
  -ltuning_rCache -lkernel_autoTuner -lglobal_thread -lnuma_manager -lpool_manager \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -lconfig_env \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lthread_scheduler \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o advanced/kFusion_engine/kernel_fusion
```
