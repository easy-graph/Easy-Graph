#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object cpp_LPA(py::object G);
