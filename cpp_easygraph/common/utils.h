#pragma once
#include "common.h"

py::object attr_to_dict(const node_attr_dict_factory& attr);
std::string weight_to_string(py::object weight);
py::object py_sum(py::object o);