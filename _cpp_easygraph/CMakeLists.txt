cmake_minimum_required(VERSION 3.15)
if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "cpp_easygraph_standalone")
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE EASYGRAPH_SOURCES "src/*.cpp")


pybind11_add_module(cpp_easygraph MODULE ${EASYGRAPH_SOURCES})

target_include_directories(cpp_easygraph PUBLIC "include")

if(SKBUILD)
    install(TARGETS cpp_easygraph DESTINATION ${SKBUILD_PROJECT_NAME})
endif ()
