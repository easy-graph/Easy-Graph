cmake_minimum_required(VERSION 3.23)
project(cpp_easygraph)
set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCES
    classes/*.cpp
    common/*.cpp
    functions/*/*.cpp
    cpp_easygraph.cpp
)

add_subdirectory(pybind11)

option(EASYGRAPH_ENABLE_GPU "EASYGRAPH_ENABLE_GPU" OFF)

# 启用OpenMP支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. Parallel optimization will be disabled.")
endif()

# 查找ARPACK库（兼容无CMake配置的系统）
find_library(ARPACK_LIB NAMES arpack arpack_double PATHS ENV LD_LIBRARY_PATH)
if(NOT ARPACK_LIB)
    message(FATAL_ERROR "ARPACK library not found")
else()
    message(STATUS "Found ARPACK library: ${ARPACK_LIB}")
endif()

find_package(LAPACK REQUIRED)

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

find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
    add_definitions(-DEIGEN_MAJOR_VERSION=${EIGEN3_VERSION_MAJOR})
endif()

# 将OpenMP链接到target
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpp_easygraph PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(cpp_easygraph PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP enabled for cpp_easygraph")
endif()

# 链接ARPACK和LAPACK
target_link_libraries(cpp_easygraph PRIVATE ${ARPACK_LIB} LAPACK::LAPACK)