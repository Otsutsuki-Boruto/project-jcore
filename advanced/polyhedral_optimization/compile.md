# **Compile**

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
  -Icomposite/dispatch_kernel/include \
  -Icomposite/tuning_rCache/include \
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
