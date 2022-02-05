#pragma once
#include <fstream>
#include "Common.h"
#include "GraphMap.h"
#include "GraphEdges.h"
#include "ReadFile.h"

union edge_tuple {
    int point[2];
    long long val;
};

struct Graph {
    PyObject_HEAD
        PyObject* graph;
    std::unordered_map<int, std::map<std::string, float>>node;
    std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>> adj;
    PyObject* node_to_id, * id_to_node;
    int id;
};

PyObject* Graph_get_graph(Graph* self, void*);

PyObject* Graph_get_nodes(Graph* self, void*);

PyObject* Graph_get_adj(Graph* self, void*);

PyObject* Graph_get_edges(Graph* self, void*);

//以下是类的方法
void _add_one_node(Graph* self, PyObject* one_node_for_adding, PyObject* node_attr, std::map<std::string, float>* c_node_attr = nullptr);

PyObject* Graph_add_node(Graph* self, PyObject* arg, PyObject* kwarg);

PyObject* Graph_add_nodes(Graph* self, PyObject* args, PyObject* kwargs);

void _add_one_edge(Graph* self, PyObject* u, PyObject* v, PyObject* edge_attr, std::map<std::string, float>* c_edge_attr = nullptr);

PyObject* Graph_add_edge(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_add_weighted_edge(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_add_edges(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_add_edges_from_file(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_degree(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_size(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_neighbors(Graph* self, PyObject* args, PyObject* kwargs);

void _remove_one_node(Graph* self, PyObject* node_to_remove);

PyObject* Graph_remove_node(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_remove_nodes(Graph* self, PyObject* args, PyObject* kwargs);

void _remove_one_edge(Graph* self, int u, int v);

PyObject* Graph_remove_edge(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_remove_edges(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_has_node(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_has_edge(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_number_of_nodes(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_number_of_edges(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_is_directed(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_copy(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_nodes_subgraph(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_ego_subgraph(Graph* self, PyObject* args, PyObject* kwargs);

PyObject* Graph_to_index_node_graph(Graph* self, PyObject* args, PyObject* kwargs);

//以下是作为sequence的方法
Py_ssize_t Graph_len(Graph* self);

int Graph_contains(Graph* self, PyObject* node);

//以下是作为mapping的方法
PyObject* Graph_getitem(Graph* self, PyObject* pykey);

//以下是类的内置方法
PyObject* Graph_new(PyTypeObject* type, PyObject* args, PyObject* kwds);

void* Graph_dealloc(PyObject* obj);

PyObject* Graph_iter(Graph* self);
