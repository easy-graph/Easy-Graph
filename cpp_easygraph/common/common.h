#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <map>
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

// The purpose of this function is to provide a way to destroy memory allocated in its scope
typedef void easygraph_finally_func_t (void*);
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



typedef struct s_vector_ptr {
    void** stor_begin;
    void** stor_end;
    void** end;
	// used for 
    easygraph_finally_func_t* item_destructor;
} igraph_vector_ptr_t;