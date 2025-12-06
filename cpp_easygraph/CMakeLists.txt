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

# Find OpenMP (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "OpenMP not found; continuing without OpenMP")
endif()

# Find ARPACK (optional; do not fail if absent)
find_library(ARPACK_LIB NAMES arpack arpack_double)
if(ARPACK_LIB)
    message(STATUS "Found ARPACK library: ${ARPACK_LIB}")
    add_compile_definitions(HAVE_ARPACK=1)
else()
    message(WARNING "ARPACK not found; ARPACK-based solvers will be disabled (fallback used).")
    add_compile_definitions(HAVE_ARPACK=0)
endif()

find_package(LAPACK REQUIRED)

# Find Eigen (optional). Change to REQUIRED if Eigen must be enforced.
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${Eigen3_INCLUDE_DIRS}")
    add_compile_definitions(HAVE_EIGEN=1)
else()
    message(STATUS "Eigen3 not found; eigen-accelerated code will be disabled")
    add_compile_definitions(HAVE_EIGEN=0)
endif()

# Link/include to cpp_easygraph only if the target exists.
# This avoids errors when the target is created elsewhere or later.
if(TARGET cpp_easygraph)
    if(Eigen3_FOUND)
        target_include_directories(cpp_easygraph PRIVATE ${Eigen3_INCLUDE_DIRS})
        if(TARGET Eigen3::Eigen)
            target_link_libraries(cpp_easygraph PRIVATE Eigen3::Eigen)
        endif()
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(cpp_easygraph PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(ARPACK_LIB)
        target_link_libraries(cpp_easygraph PRIVATE ${ARPACK_LIB} LAPACK::LAPACK)
    else()
        target_link_libraries(cpp_easygraph PRIVATE LAPACK::LAPACK)
    endif()
else()
    message(STATUS "cpp_easygraph target not defined yet; optional libs will be linked after target creation.")
endif()