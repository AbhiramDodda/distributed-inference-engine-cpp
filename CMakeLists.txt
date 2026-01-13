cmake_minimum_required(VERSION 3.15)
project(DistributedInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Find packages
find_package(Threads REQUIRED)

# ONNX Runtime (CUDA version from pacman)
set(ONNXRUNTIME_INCLUDE_DIR "/usr/include/onnxruntime" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIB_DIR "/usr/lib" CACHE PATH "ONNX Runtime library directory")

# Find CUDA (optional, for CUDA provider)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(STATUS "CUDA not found, GPU support may be limited")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/cpp-httplib
    ${CMAKE_SOURCE_DIR}/external/json/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link directories
link_directories(${ONNXRUNTIME_LIB_DIR})

# Source files
set(COMMON_SOURCES
    src/consistent_hash.cpp
    src/circuit_breaker.cpp
    src/inference_engine.cpp
)

# Note: batch_processor.h is header-only (template)

# Worker Node executable
add_executable(worker_node
    src/worker_node.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(worker_node
    Threads::Threads
    onnxruntime
)

# Gateway executable
add_executable(gateway
    src/gateway.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(gateway
    Threads::Threads
    onnxruntime
)

# Compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(worker_node PRIVATE -O3 -march=native)
    target_compile_options(gateway PRIVATE -O3 -march=native)
else()
    target_compile_options(worker_node PRIVATE -Wall -Wextra)
    target_compile_options(gateway PRIVATE -Wall -Wextra)
endif()