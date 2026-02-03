# Compile

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
  -I/usr/lib/llvm-20/include \
  -Iadvanced/neural_layer/include \
  -Iadvanced/polyhedral_optimization/include \
  -Iadvanced/jit_kernel/include \
  -Iadvanced/graph_execution/include \
  -Iadvanced/operator_graph/include \
  -Iadvanced/kFusion_engine/include \
  -Iadvanced/include/ \
  -Icomposite/mkernel_interface/include \
  -Icomposite/tuning_rCache/include \
  -Icomposite/vmath_engine/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/global_thread/include \
  -Icomposite/numa_manager/include \
  -Icomposite/pool_manager/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/mem_prefetch/include \
  -Ibase/mem_allocator/include \
  -Ibase/cache_block/include \
  -Ibase/isa_aware/include \
  -Ibase/hw_introspect/include \
  -Ibase/cpu_detect/include \
  -Ibase/config_env/include \
  advanced/neural_layer/src/*.cpp \
  -Ladvanced/lib/static \
 -lpolyhedral_optimization -ljit_kernel_generator -lgraph_execution_engine -loperator_graph -lkernel_fusion \
  -lmkernel_interface -ltuning_rCache -lvector_math_engine -ldispatch_kernel \
  -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager -lpool_manager \
  -lmicro_timer -lthread_scheduler -lffm -lffm_cache_block -lffm_prefetch -lffm_huge -lffm_prefetch \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
  -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -ltbb -lpapi -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o advanced/neural_layer/neural_layer
```