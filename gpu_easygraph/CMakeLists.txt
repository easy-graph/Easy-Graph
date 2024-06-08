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

target_include_directories(gpu_easygraph
    PRIVATE common
    PRIVATE functions
)

set_target_properties(gpu_easygraph  PROPERTIES
    LINK_SEARCH_START_STATIC ON
    LINK_SEARCH_END_STATIC ON
)