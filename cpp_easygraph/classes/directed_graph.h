#pragma once

#include "../common/common.h"
#include "graph.h"

struct DiGraph : public Graph {
    DiGraph();
    py::object get_edges();
    py::object get_pred();

    adj_dict_factory pred;
};

py::object DiGraph__init__(py::args args, py::kwargs kwargs);
py::object DiGraph_out_degree(py::object self, py::object weight);
py::object DiGraph_in_degree(py::object self, py::object weight);
py::object DiGraph_degree(py::object self, py::object weight);
py::object DiGraph_size(py::object self, py::object weight);
py::object DiGraph_number_of_edges(py::object self, py::object u, py::object v);
py::object DiGraph_predecessors(py::object self, py::object node);
py::object DiGraph_add_node(py::args args, py::kwargs kwargs);
py::object DiGraph_add_nodes(DiGraph& self, py::list nodes_for_adding, py::list nodes_attr);
py::object DiGraph_add_nodes_from(py::args args, py::kwargs kwargs);
py::object DiGraph_remove_node(DiGraph& self, py::object node_to_remove);
py::object DiGraph_remove_nodes(py::object self, py::list nodes_to_remove);
py::object DiGraph_add_edge(py::args args, py::kwargs kwargs);
py::object DiGraph_add_edges(DiGraph& self, py::list edges_for_adding, py::list edges_attr);
py::object DiGraph_add_edges_from_file(DiGraph& self, py::str file, py::object weighted, py::object is_transform);
py::object DiGraph_add_edges_from(py::args args, py::kwargs attr);
py::object DiGraph_remove_edge(DiGraph& self, py::object u, py::object v);
py::object DiGraph_remove_edges(py::object self, py::list edges_to_remove);
py::object DiGraph_remove_edges_from(py::object self, py::list ebunch);
py::object DiGraph_add_weighted_edge(DiGraph& self, py::object u_of_edge, py::object v_of_edge, weight_t weight);
py::object DiGraph_nodes_subgraph(py::object self, py::list from_nodes);
py::object DiGraph_is_directed(py::object self);
py::object DiGraph_py(py::object self);
py::object DiGraph_generate_linkgraph(py::object self, py::object weight);