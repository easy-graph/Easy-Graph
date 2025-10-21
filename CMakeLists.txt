cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cpp_easygraph MODULE src/cpp_easygraph/cpp_easygraph.cpp)

install(TARGETS cpp_easygraph DESTINATION ${SKBUILD_PROJECT_NAME})
