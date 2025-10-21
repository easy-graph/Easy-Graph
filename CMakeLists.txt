cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

cmake_policy(SET CMP0167 NEW)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS python3)

pybind11_add_module(cpp_easygraph MODULE src/cpp_easygraph/cpp_easygraph.cpp)

target_include_directories(cpp_easygraph PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(cpp_easygraph PRIVATE ${Boost_LIBRARIES})

install(TARGETS cpp_easygraph DESTINATION ${SKBUILD_PROJECT_NAME})
