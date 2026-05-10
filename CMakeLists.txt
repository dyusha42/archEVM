cmake_minimum_required(VERSION 3.16)
project(neon_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(IS_ARM FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|armv7|armv8|arm)")
    set(IS_ARM TRUE)
    message(STATUS "[neon_benchmark] Architecture: ARM (${CMAKE_SYSTEM_PROCESSOR}) — NEON enabled")
else()
    message(STATUS "[neon_benchmark] Architecture: ${CMAKE_SYSTEM_PROCESSOR} — scalar fallback")
endif()

if(MSVC)
    add_compile_options(/W4 /permissive- /O2 /Qvec-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    add_compile_options(-fno-tree-vectorize)
    if(IS_ARM AND CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
        add_compile_options(-mfpu=neon -mfloat-abi=hard)
    endif()
endif()

add_executable(neon_benchmark neon_array_improved.cpp)

set_target_properties(neon_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
