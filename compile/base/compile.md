# **Compile**

The notebook is an essential guide to create shared and static library components in base directory.

## CPU Feature Detection Module

```bash
# static library
g++ -std=c++20 -O3 -Iinclude -march=native -c src/*.cpp
ar rcs libcpu_detect.a *.o

# shared library
g++ -std=c++20 -O3 -fPIC -Iinclude -march=native -shared src/*.cpp -o libcpu_detect.so -lnuma
```

## ISA-Aware Dispatch Mechanism

```bash
# static library
gcc -std=c11 -Iinclude src/*.c -O3 -pthread -mavx -c
ar rcs libjcore_isa_aware.a *.o
rm -f *.o

# shared library
gcc -std=c11 -Iinclude src/*.c -O3 -pthread -mavx -shared -fPIC -o libjcore_isa_aware.so
```

## Hardware Introspection Layer

```bash
# static library
gcc -std=gnu11 -O3 -mavx -Wall -Wextra \
  -Iinclude \
  -c src/*.c
ar rcs libjcore_hw_introspect.a *.o


# shared library
gcc -std=gnu11 -O3 -fPIC -mavx -Wall -Wextra \
  -Iinclude \
  -c src/*.c
gcc -shared -o libjcore_hw_introspect.so *.o -lhwloc -lnuma
```

## Memory Allocator Wrapper

```bash
# static library
# Step 1: Compile object files
gcc -std=c11 -O3 -mavx -Wall -Wextra -Werror -DUSE_JEMALLOC -DUSE_MEMKIND -DUSE_NUMA \
  -Iinclude -c src/*.c -fPIC

# Step 2: Create static library
ar rcs libffm.a *.o

# shared library

gcc -std=c11 -O3 -mavx -Wall -Wextra -Werror -DUSE_JEMALLOC -DUSE_MEMKIND -DUSE_NUMA \
  -Iinclude -fPIC src/*.c -shared -ldl -lnuma -lmemkind -ljemalloc -o libffm.so
```

## Huge Page Controller

```bash
# static library
gcc -std=c11 -O3 -mavx -Wall -Wextra -Iinclude -c src/ffm_hugepage.c -o ffm_hugepage.o
ar rcs libffm_huge.a ffm_hugepage.o

# shared library
gcc -std=c11 -O3 -mavx -Wall -Wextra -fPIC -Iinclude -c src/ffm_hugepage.c -o ffm_hugepage.o
gcc -shared -o libffm_huge.so ffm_hugepage.o
```

## Cache Blocking / Tiling Utility

```bash
# static library
gcc -std=c11 -O3 -mavx -Wall -Wextra -Iinclude -c src/ffm_cache_block.c -o ffm_cache_block.o
ar rcs libffm_cache_block.a ffm_cache_block.o


# shared library
gcc -std=c11 -O3 -mavx -Wall -Wextra -fPIC -Iinclude -c src/ffm_cache_block.c -o ffm_cache_block.o
gcc -shared -o libffm_cache_block.so ffm_cache_block.o -lm

```

## Base Thread Scheduler Abstraction

```bash
# static library
# Step 1: Compile source files into object files
g++ -std=c++20 -O3 -mavx -Iinclude -c src/thread_scheduler.cpp -fopenmp

# Step 2: Create static library from object file
ar rcs libthread_scheduler.a thread_scheduler.o


# shared library
g++ -std=c++20 -O3 -mavx -Iinclude -fPIC -shared src/thread_scheduler.cpp -ltbb -fopenmp -o libthread_scheduler.so

```

## Microbenchmark & Timer Utilities

```bash
# static library
# Compile all source files in src/ into object files
g++ -std=c++20 -O3 -mavx -Iinclude -c src/* -fopenmp

# Create static library from all object files
ar rcs libmicro_timer.a *.o

# shared library
# Compile and create shared library from all sources
g++ -std=c++20 -O3 -mavx -Iinclude -fPIC -shared src/* -ltbb -fopenmp -o libmicro_timer.so
```

## Configuration & Env Controller

```bash
# static library
# Compile all sources into object files
g++ -std=c++20 -O3 -mavx -Iinclude -c src/* -pthread

# Create static library
ar rcs libconfig_env.a *.o

# shared library
# Compile and create shared library
g++ -std=c++20 -O3 -mavx -Iinclude -fPIC -shared src/* -ltbb -fopenmp -pthread -o libconfig_env.so

```

## Memory Prefetch Interface

```bash
# static library
gcc -std=c11 -O3 -mavx -Wall -Wextra -Iinclude -I../cache_block/include -c src/ffm_prefetch.c -o ffm_prefetch.o


# shared library
gcc -std=c11 -O3 -mavx -Wall -Wextra -fPIC -Iinclude -I../cache_block/include -c src/*.c -o ffm_prefetch.o
gcc -shared -o libffm_prefetch.so ffm_prefetch.o
```
