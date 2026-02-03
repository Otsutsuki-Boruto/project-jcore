# **Compile Numa Memory Manager**

```bash
g++ -std=c++20 -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC \
  -Icomposite/numa_manager/include \
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
  composite/numa_manager/src/numa_memory_manager.cpp \
  composite/numa_manager/src/test_numa_memory_manager.cpp \
  -Lbase/mem_allocator \
  -Lcomposite/lib/static \
  -Lcomposite/lib/shared \
  -lffm -lffm_huge -lffm_prefetch -lffm_cache_block \
  -ljcore_isa_aware -ljcore_hw_introspect -lcpu_detect \
  -lconfig_env -lmicro_timer -lthread_scheduler \
  -lnuma -lmemkind -ljemalloc \
  -ltbb -lpthread -ldl -lm \
  -Wl,-rpath=$(pwd)/composite/lib/shared \
  -o composite/numa_manager/numa_manager_testlpthread
```