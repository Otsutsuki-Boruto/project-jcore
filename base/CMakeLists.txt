file(GLOB_RECURSE BASE_SRC
        *.c
        *.cpp
)

add_library(jcore_base STATIC ${BASE_SRC})

target_include_directories(jcore_base
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cache_block/include
        ${CMAKE_CURRENT_SOURCE_DIR}/config_env/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_detect/include
        ${CMAKE_CURRENT_SOURCE_DIR}/hp_controller/include
        ${CMAKE_CURRENT_SOURCE_DIR}/hw_introspect/include
        ${CMAKE_CURRENT_SOURCE_DIR}/isa_aware/include
        ${CMAKE_CURRENT_SOURCE_DIR}/mem_allocator/include
        ${CMAKE_CURRENT_SOURCE_DIR}/mem_prefetch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/thread_schedule/include

)

target_link_libraries(jcore_base
        PUBLIC
        pthread
        numa
        hwloc
        dl
        memkind
        jemalloc
        tbb
)

