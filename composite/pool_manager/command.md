# **compilation command**

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC  \
  -Icomposite/pool_manager/include \
  -Icomposite/include \
  -Ibase/benchmark/include \
  -Ibase/hp_controller/include \
  -Ibase/thread_schedule/include \
  -Ibase/cpu_detect/include \
  -Ibase/mem_allocator/include \
  -Ibase/isa_aware/include \
  -Ibase/cache_block/include \
  -Ibase/config_env/include \
  -Ibase/hw_introspect/include \
  composite/pool_manager/src/*.cpp \
  -Lbase/mem_allocator \
  -Lcomposite/lib/static \
  -Lcomposite/lib/shared \
  -lffm -lffm_huge -ljcore_hw_introspect -lconfig_env \
  -lffm_prefetch -lffm_cache_block -ljcore_isa_aware \
  -lcpu_detect -lmicro_timer -lthread_scheduler \
  -lnuma -lmemkind -ljemalloc \
  -ltbb -lpthread -ldl -lm \
  -Wl,-rpath=$(pwd)/composite/lib/shared \
  -o composite/pool_manager/pool_manager_test
```

---
