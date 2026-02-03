cmake_minimum_required(VERSION 3.22)

project(JCore
        LANGUAGES C CXX
)

# ---- Global policy ----
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Build type default ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Third-party deps (prebuilt) ----
add_library(BLIS SHARED IMPORTED)
set_target_properties(BLIS PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/composite/lib/shared/libblis.so"
)

add_library(OpenBLAS SHARED IMPORTED)
set_target_properties(OpenBLAS PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/composite/lib/shared/libopenblas.so"
)

add_library(LIBXSMM SHARED IMPORTED)
set_target_properties(LIBXSMM PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/composite/lib/shared/libxsmm.so"
)


# ---- Subsystems ----
add_subdirectory(base)
add_subdirectory(composite)
add_subdirectory(advanced)
add_subdirectory(project_test)
