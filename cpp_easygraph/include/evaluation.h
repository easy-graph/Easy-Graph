#pragma once
#include "common.h"
#include "graph.h"

py::dict constraint(const py::object &G, py::object nodes,
						   const py::object &weight,
						   const py::object &n_workers /*未使用但保留签名*/);
py::object effective_size(py::object G, py::object nodes, py::object weight,
                          py::object n_workers);


py::dict hierarchy(Graph& G,
				   py::object nodes   = py::none(),
				   py::object weight  = py::none(),
				   py::object n_workers = py::none());
