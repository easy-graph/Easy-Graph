#pragma once

#include "../../common/common.h"

py::object constraint(py::object G, py::object nodes, py::object weight, py::object n_workers);
py::object effective_size(py::object G, py::object nodes, py::object weight, py::object n_workers);
py::object efficiency(py::object G, py::object nodes, py::object weight, py::object n_workers);
py::object hierarchy(py::object G, py::object nodes, py::object weight, py::object n_workers);