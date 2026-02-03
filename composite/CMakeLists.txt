file(GLOB_RECURSE COMPOSITE_SRC
        *.c
        *.cpp
)

add_library(jcore_composite STATIC ${COMPOSITE_SRC}
        mkernel_interface/src/mil_vector.cpp)

target_include_directories(jcore_composite
        PUBLIC
        ../compile
        /home/muhammad-awais/sleef/build/include
        /home/muhammad-awais/sleef/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_kernel/include
        ${CMAKE_CURRENT_SOURCE_DIR}/global_thread/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_autoTuner/include
        ${CMAKE_CURRENT_SOURCE_DIR}/mkernel_interface/include
        ${CMAKE_CURRENT_SOURCE_DIR}/numa_manager/include
        ${CMAKE_CURRENT_SOURCE_DIR}/perf_profiler/include
        ${CMAKE_CURRENT_SOURCE_DIR}/pool_manager/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tuning_rCache/include
        ${CMAKE_CURRENT_SOURCE_DIR}/vmath_engine/include
)

target_link_libraries(jcore_composite
        PUBLIC
        jcore_base
        BLIS
        OpenBLAS
        SLEEF
        papi

)
