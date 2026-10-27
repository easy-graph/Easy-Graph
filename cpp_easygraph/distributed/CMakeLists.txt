cmake_minimum_required(VERSION 3.23)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(easygraph_distributed LANGUAGES CXX)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/EasyGraphOpenMP.cmake")
find_package(MPI REQUIRED COMPONENTS CXX)
pybind11_add_module(cpp_easygraph_distributed
    bindings.cpp
    communicator.cpp
    distributed_graph.cpp
    partitioner.cpp
    loader.cpp
    worker.cpp
    algorithms/pagerank.cpp
)
target_compile_features(cpp_easygraph_distributed PRIVATE cxx_std_11)
target_link_libraries(cpp_easygraph_distributed PRIVATE MPI::MPI_CXX easygraph_openmp)
