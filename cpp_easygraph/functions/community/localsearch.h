#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object cpp_localsearch(
    py::object G,
    py::object center_num = py::none(),
    py::object auto_choose_centers = py::bool_(false),
    py::object maximum_tree = py::bool_(true),
    py::object seed = py::none(),
    py::object self_loop = py::bool_(false)
);
