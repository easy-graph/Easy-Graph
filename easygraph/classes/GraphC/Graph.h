#pragma once
#define BOOST_PYTHON_STATIC_LIB

#include "Common.h"

extern py::object MappingProxyType;

struct Graph
{
	typedef int node_t;
	typedef float weight_t;
	typedef std::map<std::string, weight_t> node_attr_dict_factory; //(weight_key, value)
	typedef std::map<std::string, weight_t> edge_attr_dict_factory; //(weight_key, value)
	typedef std::unordered_map<node_t, node_attr_dict_factory> node_dict_factory; //(node, node_attr)
	typedef std::unordered_map<node_t, edge_attr_dict_factory> adj_attr_dict_factory; //(out_node, (weight_key, value))
	typedef std::unordered_map<node_t, adj_attr_dict_factory> adj_dict_factory; //(node, edge_attr)

	node_dict_factory node;
	adj_dict_factory adj;
	py::dict node_to_id, id_to_node, graph;
	node_t id;
	bool dirty_nodes, dirty_adj;
	py::object nodes_cache, adj_cache;

	Graph();
	py::object get_nodes();
	py::object get_name();
	py::object get_graph();
	py::object get_adj();
	py::object get_edges();
};

py::object __init__(py::tuple args, py::dict kwargs);
py::object __iter__(py::object self);
py::object __len__(py::object self);
py::object __contains__(py::object self, py::object node);
py::object __getitem__(py::object self, py::object node);
py::object add_node(py::tuple args, py::dict kwargs);
py::object add_nodes(Graph& self, py::list nodes_for_adding, py::list nodes_attr);
py::object add_nodes_from(py::tuple args, py::dict kwargs);
py::object remove_node(Graph& self, py::object node_to_remove);
py::object remove_nodes(py::object self, py::list nodes_to_remove);
py::object number_of_nodes(Graph& self);
py::object has_node(Graph& self, py::object node);
py::object add_edge(py::tuple args, py::dict kwargs);
py::object add_edges(Graph& self, py::list edges_for_adding, py::list edges_attr);
py::object add_edges_from(py::tuple args, py::dict attr);
py::object add_edges_from_file(Graph& self, py::str file, py::object weighted);
py::object add_weighted_edge(Graph& self, py::object u_of_edge, py::object v_of_edge, Graph::weight_t weight);
py::object remove_edge(Graph& self, py::object u, py::object v);
py::object remove_edges(py::object self, py::list edges_to_remove);
py::object number_of_edges(py::object self);
py::object has_edge(Graph& self, py::object u, py::object v);
py::object copy(py::object self);
py::object degree(py::object self, py::object weight);
py::object neighbors(py::object self, py::object node);
py::object nodes_subgraph(py::object self, py::list from_nodes);
py::object ego_subgraph(py::object self, py::object center);
py::object size(py::object self, py::object weight);
py::object is_directed(py::object self);
py::object is_multigraph(py::object self);

