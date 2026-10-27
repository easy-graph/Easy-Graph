cmake_minimum_required(VERSION 3.23)
project(cpp_easygraph)
set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCES
    classes/*.cpp
    common/*.cpp
    functions/*/*.cpp
    cpp_easygraph.cpp
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pybind11/CMakeLists.txt")
    add_subdirectory(pybind11)
else()
    find_package(pybind11 CONFIG REQUIRED)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/EasyGraphOpenMP.cmake")

if(EASYGRAPH_ENABLE_DISTRIBUTED)
    add_subdirectory(distributed)
endif()

option(EASYGRAPH_ENABLE_OPENMP "Enable OpenMP acceleration (auto-detect)" ON)
option(EASYGRAPH_ENABLE_GPU "EASYGRAPH_ENABLE_GPU" OFF)

if (EASYGRAPH_ENABLE_GPU)

    pybind11_add_module(cpp_easygraph
        ${SOURCES}
        $<TARGET_OBJECTS:gpu_easygraph>
    )

    set_property(TARGET cpp_easygraph PROPERTY CUDA_ARCHITECTURES all)

    target_compile_definitions(cpp_easygraph
        PRIVATE EASYGRAPH_ENABLE_GPU
    )

    target_link_libraries(cpp_easygraph
        PRIVATE cudart_static
    )

else()
    pybind11_add_module(cpp_easygraph
        ${SOURCES}
    )

endif()

target_link_libraries(cpp_easygraph PRIVATE easygraph_openmp)
