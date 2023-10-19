cmake_minimum_required(VERSION 3.23)
project(gpu_easygraph LANGUAGES CXX CUDA)

file(GLOB SOURCES
    functions/*/*.c
    functions/*/*.cpp
    functions/*/*.cu
    common/*.c
    common/*.cpp
    gpu_easygraph.cpp
)

add_subdirectory(pybind11)
pybind11_add_module(gpu_easygraph ${SOURCES})
set_property(TARGET gpu_easygraph PROPERTY CUDA_ARCHITECTURES all)

target_include_directories(gpu_easygraph
PRIVATE common
PRIVATE functions
)
