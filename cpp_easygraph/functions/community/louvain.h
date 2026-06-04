#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;


py::object cpp_louvain_communities(
    py::object G,
    py::object weight = py::str("weight"),
    py::object threshold = py::float_(0.0),
    py::object resolution = py::float_(1.0)
);

py::object cpp_louvain_communities_serial(
    py::object G,
    py::object weight = py::str("weight"),
    py::object threshold = py::float_(0.0),
    py::object resolution = py::float_(1.0)
);