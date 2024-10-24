#pragma once

#include "../common/common.h"
#include "csr_graph.h"
#include "linkgraph.h"
#include "coo_graph.h"

// old version
struct Graph {
    node_dict_factory node;
    adj_dict_factory adj;
    Graph_L linkgraph_structure;
    std::shared_ptr<CSRGraph> csr_graph;
    py::kwargs node_to_id, id_to_node, graph;
    node_t id;
    bool dirty_nodes, dirty_adj, linkgraph_dirty;
    py::object nodes_cache, adj_cache;
    std::shared_ptr<COOGraph> coo_graph;

    Graph();
    py::object get_nodes();
    py::object get_name();
    py::object set_name(py::object name);
    py::object get_graph();
    py::object get_adj();
    py::object get_edges();
    py::object get_node_index();
    std::vector<graph_edge> _get_edges(bool if_directed=true);
    bool is_linkgraph_dirty();
    Graph_L _get_linkgraph_structure();
    void drop_cache();

    std::shared_ptr<CSRGraph> gen_CSR(const std::string& weight);
    std::shared_ptr<CSRGraph> gen_CSR();
    std::shared_ptr<std::vector<int>> gen_CSR_sources(const py::object& py_sources);
    std::shared_ptr<COOGraph> gen_COO();
    std::shared_ptr<COOGraph> gen_COO(const std::string& weight);
    std::shared_ptr<COOGraph> transfer_csr_to_coo(const std::shared_ptr<CSRGraph>& csr_graph);
};

py::object Graph__init__(py::args args, py::kwargs kwargs);
py::object Graph__iter__(py::object self);
py::object Graph__len__(py::object self);
py::object Graph__contains__(py::object self, py::object node);
py::object Graph__getitem__(py::object self, py::object node);
py::object Graph_add_node(py::args args, py::kwargs kwargs);
py::object Graph_add_nodes(Graph& self, py::list nodes_for_adding, py::list nodes_attr);
py::object Graph_add_nodes_from(py::args args, py::kwargs kwargs);
py::object Graph_remove_node(Graph& self, py::object node_to_remove);
py::object Graph_remove_nodes(py::object self, py::list nodes_to_remove);
py::object Graph_number_of_nodes(Graph& self);
py::object Graph_has_node(Graph& self, py::object node);
py::object Graph_nbunch_iter(py::object self, py::object nbunch);
py::object Graph_add_edge(py::args args, py::kwargs kwargs);
py::object Graph_add_edges(Graph& self, py::list edges_for_adding, py::list edges_attr);
py::object Graph_add_edges_from(py::args args, py::kwargs attr);
py::object Graph_add_edges_from_file(Graph& self, py::str file, py::object weighted, py::object is_transform);
py::object Graph_add_weighted_edge(Graph& self, py::object u_of_edge, py::object v_of_edge, weight_t weight);
py::object Graph_remove_edge(Graph& self, py::object u, py::object v);
py::object Graph_remove_edges(py::object self, py::list edges_to_remove);
py::object Graph_number_of_edges(py::object self, py::object u, py::object v);
py::object Graph_has_edge(Graph& self, py::object u, py::object v);
py::object Graph_copy(py::object self);
py::object Graph_degree(py::object self, py::object weight);
py::object Graph_neighbors(py::object self, py::object node);
py::object Graph_nodes_subgraph(py::object self, py::list from_nodes);
py::object Graph_ego_subgraph(py::object self, py::object center);
py::object Graph_size(py::object self, py::object weight);
py::object Graph_is_directed(py::object self);
py::object Graph_is_multigraph(py::object self);
py::object Graph_to_index_node_graph(py::object self, py::object begin_index);
py::object Graph_generate_linkgraph(py::object self, py::object weight);
py::object Graph_py(py::object self);