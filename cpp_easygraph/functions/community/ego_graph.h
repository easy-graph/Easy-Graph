#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object cpp_ego_graph(
    py::object G,
    py::object n,
    py::object radius = py::int_(1),
    py::object center = py::bool_(true),
    py::object undirected = py::bool_(false),
    py::object distance = py::none()
);

py::object cpp_ego_graph_csr(
    py::object G,
    py::object n,
    py::object radius = py::int_(1),
    py::object center = py::bool_(true),
    py::object undirected = py::bool_(false),
    py::object distance = py::none()
);
