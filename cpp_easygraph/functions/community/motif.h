#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::list cpp_enumerate_subgraph(py::object G, int k);

py::int_ cpp_count_enumerate_subgraph(py::object G, int k);

py::list cpp_random_enumerate_subgraph(py::object G, int k, py::object cut_prob);
