#pragma once

#include "common.h"

py::dict attr_to_dict(const node_attr_dict_factory& attr);
std::string weight_to_string(py::handle weight);
py::object py_sum(const py::object& o);
