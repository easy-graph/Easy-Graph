#pragma once
#define BOOST_PYTHON_STATIC_LIB

#include "Common.h"
#include "Graph.h"
py::object _dijkstra_multisource(py::object G, py::object sources, py::object weight, py::object target);
py::object Floyd(py::object G);