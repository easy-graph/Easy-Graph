#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;


py::object cpp_greedy_modularity_communities(
    py::object G, 
    py::object weight = py::str("weight")
);
