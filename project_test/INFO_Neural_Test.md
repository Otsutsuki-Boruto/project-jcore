# Compile

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE \
-Iadvanced/neural_layer/include project_test/neural_layer_test.cpp \
 -Ladvanced/lib/static/ \
 -lneural_layer -lpolyhedral_optimization -ljit_kernel_generator \
 -lgraph_execution_engine -loperator_graph -lkernel_fusion \
 -lmkernel_interface -ltuning_rCache -lvector_math_engine -ldispatch_kernel \
 -lkernel_autoTuner -lperf_profiler -lglobal_thread -lnuma_manager -lpool_manager \
 -lmicro_timer -lthread_scheduler -lffm -lffm_cache_block -lffm_prefetch -lffm_huge -lffm_prefetch \
 -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect -lconfig_env \
 -Wl,--whole-archive $(llvm-config-20 --libs all) -Wl,--no-whole-archive \
  $(llvm-config-20 --ldflags --system-libs) \
  -lopenblas -lblis \
  -ltbb -lpapi -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o project_test/neural_test
```