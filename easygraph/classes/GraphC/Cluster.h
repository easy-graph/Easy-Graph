#pragma once
#define BOOST_PYTHON_STATIC_LIB

#include "Common.h"
#include "Graph.h"

py::object clustering(py::object G, py::object nodes, py::object weight);