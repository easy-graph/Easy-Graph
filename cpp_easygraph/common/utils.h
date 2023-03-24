#pragma once
#include "common.h"
#include "../classes/linkgraph.h"
#include "../classes/graph.h"

py::object attr_to_dict(const node_attr_dict_factory& attr);
std::string weight_to_string(py::object weight);
py::object py_sum(py::object o);
Graph_L graph_to_linkgraph(Graph &G, bool if_directed=false, std::string weight_key = "weight", bool is_deg = false, bool is_reverse = false);