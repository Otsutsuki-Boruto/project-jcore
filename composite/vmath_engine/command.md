# **Compilation command**

```bash
# do not include these flags! -Wall -Wextra -march=native
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/kernel_autoTuner/include \
  -Icomposite/dispatch_kernel/include \
  -Icomposite/vmath_engine/include \
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
  composite/vmath_engine/src/*.cpp \
  -Lcomposite/lib/static \
  -Lbase/mem_allocator \
  -ldispatch_kernel -lkernel_autoTuner \
  -lthread_scheduler -lmicro_timer -lconfig_env \
  -lffm_cache_block -lffm -lffm_huge -lffm_prefetch -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lopenblas -lblis \
  -ltbb -lhwloc -lnuma -lsleef -lmemkind -ljemalloc -lpthread -ldl -lm \
  -o composite/vmath_engine/vmath_engine

```
