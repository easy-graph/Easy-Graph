#pragma once
#define BOOST_PYTHON_STATIC_LIB

#include "graph.h"
#include "../common/common.h"

struct DiGraph: public Graph
{
	DiGraph();
};

py::object DiGraph__init__(py::tuple args, py::dict kwargs);