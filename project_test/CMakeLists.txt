# test/CMakeLists.txt

file(GLOB_RECURSE TEST_SRC
        "*.cpp"
        "*.c"
)

add_executable(neural_test ${TEST_SRC})

# Include directories (add whatever your test needs)
target_include_directories(neural_test
        PUBLIC
        ${CMAKE_SOURCE_DIR}/advanced/neural_layer/include
        ${CMAKE_SOURCE_DIR}/advanced/polyhedral_optimization/include
        ${CMAKE_SOURCE_DIR}/advanced/jit_kernel/include
        ${CMAKE_SOURCE_DIR}/advanced/graph_execution/include
        ${CMAKE_SOURCE_DIR}/advanced/operator_graph/include
        ${CMAKE_SOURCE_DIR}/advanced/kFusion_engine/include
)

# Link all modules as target libraries
target_link_libraries(neural_test
        PRIVATE
        jcore_advanced
        polyhedral_optimization
        jit_kernel_generator
        graph_execution_engine
        operator_graph
        kernel_fusion
        mkernel_interface
        tuning_rCache
        vector_math_engine
        dispatch_kernel
        kernel_autoTuner
        perf_profiler
        global_thread
        numa_manager
        pool_manager
        micro_timer
        thread_scheduler
        ffm
        ffm_cache_block
        ffm_prefetch
        ffm_huge
        jcore_isa_aware
        jcore_hw_introspect
        cpu_detect
        config_env
        openblas
        blis
        tbb
        papi
        hwloc
        numa
        sleef
        memkind
        pthread
        dl
        m
)

# Compiler options
target_compile_features(neural_test PUBLIC cxx_std_20)
target_compile_options(neural_test PRIVATE -O3 -mfma -msse2 -mavx -mavx2 -fopenmp -fPIC -DEVE_NO_FORCEINLINE)

# LLVM flags (if needed)
execute_process(
        COMMAND llvm-config-20 --libs all
        OUTPUT_VARIABLE LLVM_LIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND llvm-config-20 --ldflags --system-libs
        OUTPUT_VARIABLE LLVM_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
target_link_libraries(neural_test PRIVATE ${LLVM_LIBS} ${LLVM_LDFLAGS})
