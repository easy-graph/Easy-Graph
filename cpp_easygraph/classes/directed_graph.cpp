#include "directed_graph.h"
#include "../common/utils.h"

DiGraph::DiGraph(): Graph() {

}

py::object DiGraph__init__(py::tuple args, py::dict kwargs) {
	py::object MappingProxyType = py::module_::import("types").attr("MappingProxyType");
	py::object self = args[0];
	self.attr("__init__")();
	DiGraph& self_ = self.cast<DiGraph&>();
	py::dict graph_attr = kwargs;
	self_.graph.attr("update")(graph_attr);
	self_.nodes_cache = MappingProxyType(py::dict());
	self_.adj_cache = MappingProxyType(py::dict());
	return py::object();
}

py::object DiGraph_out_degree(py::object self, py::object weight) {
	py::dict degree = py::dict();
	py::list edges = self.attr("edges").cast<py::list>();
	py::object u, v;
	py::dict d;
	for (int i = 0;i < py::len(edges);i++) {
		py::tuple edge = edges[i].cast<py::tuple>();
		u = edge[0];
		v = edge[1];
		d = edge[2].cast<py::dict>();
		if (degree.contains(u)) {
			py::object(degree[u]) += d.attr("get")(weight, 1);
		}
		else {
			degree[u] = d.attr("get")(weight, 1);
		}
	}
	py::list nodes = py::list(self.attr("nodes"));
	for (int i = 0;i < py::len(nodes);i++) {
		py::object node = nodes[i];
		if (!degree.contains(node)) {
			degree[node] = 0;
		}
	}
	return degree;
}

py::object DiGraph_in_degree(py::object self, py::object weight) {
	py::dict degree = py::dict();
	py::list edges = self.attr("edges").cast<py::list>();
	py::object u, v;
	py::dict d;
	for (int i = 0;i < py::len(edges);i++) {
		py::tuple edge = edges[i].cast<py::tuple>();
		u = edge[0];
		v = edge[1];
		d = edge[2].cast<py::dict>();
		if (degree.contains(v)) {
			py::object(degree[v]) += d.attr("get")(weight, 1);
		}
		else {
			degree[v] = d.attr("get")(weight, 1);
		}
	}
	py::list nodes = py::list(self.attr("nodes"));
	for (int i = 0;i < py::len(nodes);i++) {
		py::object node = nodes[i];
		if (!degree.contains(node)) {
			degree[node] = 0;
		}
	}
	return degree;
}

py::object DiGraph_degree(py::object self, py::object weight) {
	py::dict degree = py::dict();
	py::dict out_degree = self.attr("out_degree")(weight).cast<py::dict>();
	py::dict in_degree = self.attr("in_degree")(weight).cast<py::dict>();
	py::list nodes = py::list(self.attr("nodes"));
	for (int i = 0;i < py::len(nodes);i++) {
		py::object u = nodes[i];
		degree[u] = out_degree[u] + in_degree[u];
	}
	return degree;
}

py::object DiGraph_size(py::object self, py::object weight) {
	py::dict out_degree = self.attr("out_degree")(weight).cast<py::dict>();
	py::object s = py_sum(out_degree.attr("values")());
	return (weight.is_none()) ? py::int_(s) : s;
}

py::object DiGraph_number_of_edges(py::object self, py::object u, py::object v) {
	if (u.is_none()) {
		return self.attr("size")();
	}
	Graph& G = self.cast<Graph&>();
	node_t u_id = G.node_to_id[u].cast<node_t>();
	node_t v_id = G.node_to_id.attr("get")(v, -1).cast<node_t>();
	return py::cast(int(v.cast<node_t>() != -1 && G.adj[u_id].count(v_id)));
}