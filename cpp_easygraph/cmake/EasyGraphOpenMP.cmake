include_guard(GLOBAL)

option(EASYGRAPH_ENABLE_OPENMP "Enable OpenMP acceleration (auto-detect)" ON)

if(EASYGRAPH_ENABLE_OPENMP)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_path(EG_LIBOMP_INCLUDE_DIR NAMES omp.h
            PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
        find_library(EG_LIBOMP_LIBRARY NAMES omp libomp
            PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
        if(EG_LIBOMP_INCLUDE_DIR AND EG_LIBOMP_LIBRARY)
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_INCLUDE_DIR "${EG_LIBOMP_INCLUDE_DIR}" CACHE PATH "" FORCE)
            set(OpenMP_omp_LIBRARY "${EG_LIBOMP_LIBRARY}" CACHE FILEPATH "" FORCE)
        endif()
    endif()
    find_package(OpenMP QUIET)
endif()

add_library(easygraph_openmp INTERFACE)
if(EASYGRAPH_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(easygraph_openmp INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(easygraph_openmp INTERFACE EASYGRAPH_USE_OPENMP=1)
    message(STATUS "EasyGraph OpenMP acceleration enabled")
else()
    target_compile_definitions(easygraph_openmp INTERFACE EASYGRAPH_USE_OPENMP=0)
    message(STATUS "EasyGraph OpenMP unavailable or disabled; using one thread")
endif()
