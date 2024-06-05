#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <set>
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <vector>
#include <thread>
#include <inttypes.h>
#include <limits>
namespace py = pybind11;

typedef int node_t;
typedef float weight_t;
typedef std::map<std::string, weight_t> node_attr_dict_factory; //(weight_key, value)
typedef std::map<std::string, weight_t> edge_attr_dict_factory; //(weight_key, value)
typedef std::unordered_map<node_t, node_attr_dict_factory> node_dict_factory; //(node, node_attr)
typedef std::unordered_map<node_t, edge_attr_dict_factory> adj_attr_dict_factory; //(out_node, (weight_key, value))
typedef std::unordered_map<node_t, adj_attr_dict_factory> adj_dict_factory; //(node, edge_attr)
struct graph_edge {
	node_t u, v;
	edge_attr_dict_factory attr;
	graph_edge(node_t, node_t, edge_attr_dict_factory);
};

