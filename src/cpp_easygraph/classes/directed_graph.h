#pragma once
#define BOOST_PYTHON_STATIC_LIB

#include "graph.h"
#include "../common/common.h"

struct DiGraph: public Graph
{
	DiGraph();
};

py::object DiGraph__init__(DiGraph *self, py::args args, py::kwargs kwargs);
py::object DiGraph_out_degree(DiGraph &self, py::object weight);
py::object DiGraph_in_degree(DiGraph &self, py::object weight);
py::object DiGraph_degree(DiGraph &self, py::object weight);
py::object DiGraph_size(DiGraph &self, py::object weight);
py::object DiGraph_number_of_edges(DiGraph &self, py::object u, py::object v);
