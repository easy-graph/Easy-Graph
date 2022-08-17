#pragma once

#include "graph.h"
#include "../common/common.h"

struct DiGraph: public Graph
{
	DiGraph();
};

py::object DiGraph__init__(py::tuple args, py::dict kwargs);

py::object DiGraph_out_degree(py::object self, py::object weight);
py::object DiGraph_in_degree(py::object self, py::object weight);
py::object DiGraph_degree(py::object self, py::object weight);
py::object DiGraph_size(py::object self, py::object weight);
py::object DiGraph_number_of_edges(py::object self, py::object u, py::object v);