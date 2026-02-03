file(GLOB_RECURSE ADVANCED_SRC
        *.c
        *.cpp
)

add_library(jcore_advanced STATIC ${ADVANCED_SRC}
        neural_layer/include/neural_primitives.h
        neural_layer/include/neural_primitives_internal.h
        neural_layer/src/npl_activation_norm.cpp
        neural_layer/src/npl_convolution.cpp
        neural_layer/src/npl_tensor_ops.cpp
        neural_layer/src/npl_core.cpp
        neural_layer/src/npl_pooling.cpp
        neural_layer/src/npl_test.cpp)

target_include_directories(jcore_advanced
        PUBLIC
        /opt/intel/oneapi/mkl/2025.3/include
        /usr/include/llvm-20/
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_execution/include
        ${CMAKE_CURRENT_SOURCE_DIR}/neural_layer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/jit_kernel/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kFusion_engine/include
        ${CMAKE_CURRENT_SOURCE_DIR}/operator_graph/include
        ${CMAKE_CURRENT_SOURCE_DIR}/polyhedral_optimization/include
)

target_link_libraries(jcore_advanced
        PUBLIC
        jcore_composite
)
