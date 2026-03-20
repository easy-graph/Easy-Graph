#pragma once

#include "../../common/common.h"
#include <pybind11/numpy.h>

py::object clustering(py::object G, py::object nodes, py::object weight);
double     cpp_average_clustering(py::object G);
py::tuple  cpp_clustering_array(py::object G);