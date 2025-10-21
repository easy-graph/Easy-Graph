#pragma once


#include "common.h"
#include "graph.h"

py::dict _dijkstra_multisource(const py::object &G,
									  const py::object &sources,
									  const py::object &weight,
									  const py::object &target);
py::dict Floyd(Graph& G);
py::dict Prim(Graph& G);
py::dict Kruskal(Graph& G);
