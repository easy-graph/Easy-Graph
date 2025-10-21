#pragma once


#include "common.h"

struct Graph
{
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

void Graph__init__(Graph &self, py::kwargs kwargs);
py::object Graph__iter__(py::object self);
py::object Graph__len__(const py::object& self);
py::object Graph__contains__(const py::object& self, const py::object& node);
py::object Graph__getitem__(py::object self, py::object node);
py::object Graph_add_node(const py::tuple& args, const py::dict& kwargs);
py::object Graph_add_nodes(Graph& self,const py::list& nodes_for_adding,const py::list& nodes_attr);
void Graph_add_nodes_from(Graph& self, const py::iterable& nodes_for_adding, const py::kwargs& kwargs);
void Graph_remove_node(Graph& self, const py::object& node_to_remove);
void Graph_remove_nodes(Graph& self, const py::sequence& nodes_to_remove);
py::object Graph_number_of_nodes(Graph& self);
bool Graph_has_node(Graph &self, py::object node);
py::object Graph_nbunch_iter(py::object self, py::object nbunch);
void Graph_add_edge(Graph& self, const py::object& u_of_edge, const py::object& v_of_edge, const py::kwargs& kwargs);
void Graph_add_edges(Graph& self, const py::sequence& edges_for_adding, const py::sequence& edges_attr) ;
void Graph_add_edges_from(Graph& self, const py::iterable& ebunch, const py::kwargs& attr);
void Graph_add_edges_from_file(Graph& self, const py::str& file, bool weighted);
py::object Graph_add_weighted_edge(Graph& self, py::object u_of_edge, py::object v_of_edge, weight_t weight);
void Graph_remove_edge(Graph& self, const py::object& u, const py::object& v);
void Graph_remove_edges(Graph& self, const py::sequence& edges_to_remove);
int Graph_number_of_edges(const Graph& self, py::object u = py::none(), py::object v = py::none());
bool Graph_has_edge(const Graph& self, const py::object& u, const py::object& v);
py::object Graph_copy(py::handle self_h);
py::dict Graph_degree(const Graph& self, py::object weight = py::none());
py::list Graph_neighbors(const Graph& self, const py::object& node);
py::object Graph_nodes_subgraph(py::object self, py::list from_nodes);
py::object Graph_ego_subgraph(py::object self, py::object center);
py::object Graph_size(const Graph& G, py::object weight = py::none());
bool Graph_is_directed(const Graph& self);
bool Graph_is_multigraph(const Graph& self);
