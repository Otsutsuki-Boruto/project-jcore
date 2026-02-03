# **Compile**

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
