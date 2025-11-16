cmake_minimum_required(VERSION 3.23)
project(cpp_easygraph)
set(CMAKE_CXX_STANDARD 11)


find_package(OpenMP REQUIRED)

file(GLOB SOURCES
    classes/*.cpp
    common/*.cpp
    functions/*/*.cpp
    cpp_easygraph.cpp
)

add_subdirectory(pybind11)

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


target_link_libraries(cpp_easygraph PRIVATE OpenMP::OpenMP_CXX)


#set_target_properties(cpp_easygraph  PROPERTIES
#    LINK_SEARCH_START_STATIC ON
#    LINK_SEARCH_END_STATIC ON
#)