#pragma once
#define BOOST_PYTHON_STATIC_LIB

#include "Common.h"
#include "Graph.h"

py::object attr_to_dict(const Graph::node_attr_dict_factory& attr);
std::string weight_to_string(py::object weight);

extern py::object warn;