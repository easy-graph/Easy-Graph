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

if (EASYGRAPH_ENABLE_OPENMP)
    if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_path(EG_LIBOMP_INCLUDE_DIR
            NAMES omp.h
            PATHS
                /opt/homebrew/opt/libomp/include
                /usr/local/opt/libomp/include
        )

        find_library(EG_LIBOMP_LIBRARY
            NAMES omp libomp
            PATHS
                /opt/homebrew/opt/libomp/lib
                /usr/local/opt/libomp/lib
        )

        if (EG_LIBOMP_INCLUDE_DIR AND EG_LIBOMP_LIBRARY)
            message(STATUS "libomp found: ${EG_LIBOMP_LIBRARY}")

            set(OpenMP_C_FLAGS   "-Xpreprocessor -fopenmp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "" FORCE)

            set(OpenMP_CXX_INCLUDE_DIR "${EG_LIBOMP_INCLUDE_DIR}" CACHE PATH "" FORCE)
            set(OpenMP_omp_LIBRARY     "${EG_LIBOMP_LIBRARY}"     CACHE FILEPATH "" FORCE)
        else()
            message(STATUS
                "libomp not found on macOS. OpenMP will be disabled.\n"
                "To enable OpenMP, run: brew install libomp"
            )
        endif()
    endif()

    find_package(OpenMP QUIET)
endif()

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel acceleration.")
    target_link_libraries(cpp_easygraph PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(cpp_easygraph PRIVATE EASYGRAPH_USE_OPENMP=1)
else()
    message(STATUS "OpenMP not found, building in single-thread mode.")
    target_compile_definitions(cpp_easygraph PRIVATE EASYGRAPH_USE_OPENMP=0)
endif()