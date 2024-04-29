cmake_minimum_required(VERSION 3.23)
project(gpu_easygraph LANGUAGES CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCES
    functions/*/*.c
    functions/*/*.cpp
    functions/*/*.cu
    common/*.c
    common/*.cpp
)

add_library(gpu_easygraph OBJECT ${SOURCES})

set_property(TARGET gpu_easygraph PROPERTY CUDA_ARCHITECTURES all)

target_include_directories(gpu_easygraph
    PRIVATE common
    PRIVATE functions
)

target_link_libraries(gpu_easygraph
    PRIVATE -static-libgcc
    PRIVATE -static-libstdc++
)