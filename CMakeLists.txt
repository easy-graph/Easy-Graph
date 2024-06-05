cmake_minimum_required(VERSION 3.23)

project(easygraph)

option(EASYGRAPH_ENABLE_GPU "EASYGRAPH_ENABLE_GPU" OFF)

add_subdirectory(cpp_easygraph)

if (EASYGRAPH_ENABLE_GPU)

    message("easygraph gpu module is enabled")

    add_subdirectory(gpu_easygraph)

    target_include_directories(cpp_easygraph
        PRIVATE gpu_easygraph
    )

else()

    message("easygraph gpu module is disabled")

endif()