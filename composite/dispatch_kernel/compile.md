# **compilation**

```bash
# do not use -march=native flag which can disable some hand-tuned library-level vectorization
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  -Ibase/mem_prefetch/include \
  composite/dispatch_kernel/src/*.cpp \
  -Lcomposite/lib/static \
  -lkernel_autoTuner \
  -lmicro_timer \
  -lthread_scheduler \
  -lffm -lffm_huge -lffm_prefetch -lffm_cache_block \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lconfig_env \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/dispatch_kernel/kernel_dispatch
```
