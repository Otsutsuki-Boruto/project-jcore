# **Compile**

```bash
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
