#pragma once


#include "graph.h"
#include "common.h"

struct DiGraph: public Graph
{
	DiGraph();
};

void DiGraph__init__(DiGraph &self, py::kwargs kwargs);
py::dict DiGraph_out_degree(const py::object& self, const py::object& weight);
py::dict DiGraph_in_degree(const py::object& self, const py::object& weight);
py::dict DiGraph_degree(const py::object &self,const py::object &weight);
py::object DiGraph_size(const py::object& self, const py::object& weight);
py::object DiGraph_number_of_edges(const py::object &self,const py::object &u,const py::object &v);
