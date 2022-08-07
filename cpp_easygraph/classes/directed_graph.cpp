#include "directed_graph.h"
#include "../common/utils.h"

DiGraph::DiGraph(): Graph() {

}

py::object DiGraph__init__(py::tuple args, py::dict kwargs) {
	py::object MappingProxyType = py::import("types").attr("MappingProxyType");
	py::object self = args[0];
	self.attr("__init__")();
	DiGraph& self_ = py::extract<DiGraph&>(self);
	py::dict graph_attr = kwargs;
	self_.graph.update(graph_attr);
	self_.nodes_cache = MappingProxyType(py::dict());
	self_.adj_cache = MappingProxyType(py::dict());
	return py::object();
}

py::object DiGraph_out_degree(py::object self, py::object weight) {
	py::dict degree = py::dict();
	py::list edges = py::extract<py::list>(self.attr("edges"));
	py::object u, v;
	py::dict d;
	for (int i = 0;i < py::len(edges);i++) {
		py::tuple edge = py::extract<py::tuple>(edges[i]);
		u = edge[0];
		v = edge[1];
		d = py::extract<py::dict>(edge[2]);
		if (degree.contains(u)) {
			degree[u] += d.get(weight, 1);
		}
		else {
			degree[u] = d.get(weight, 1);
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
	py::list edges = py::extract<py::list>(self.attr("edges"));
	py::object u, v;
	py::dict d;
	for (int i = 0;i < py::len(edges);i++) {
		py::tuple edge = py::extract<py::tuple>(edges[i]);
		u = edge[0];
		v = edge[1];
		d = py::extract<py::dict>(edge[2]);
		if (degree.contains(v)) {
			degree[v] += d.get(weight, 1);
		}
		else {
			degree[v] = d.get(weight, 1);
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
	py::dict out_degree = py::extract<py::dict>(self.attr("out_degree")(weight));
	py::dict in_degree = py::extract<py::dict>(self.attr("in_degree")(weight));
	py::list nodes = py::list(self.attr("nodes"));
	for (int i = 0;i < py::len(nodes);i++) {
		py::object u = nodes[i];
		degree[u] = out_degree[u] + in_degree[u];
	}
	return degree;
}

py::object DiGraph_size(py::object self, py::object weight) {
	py::dict out_degree = py::extract<py::dict>(self.attr("out_degree")(weight));
	py::object s = sum(out_degree.values());
	return (weight == py::object()) ? py::object(py::extract<int>(s)) : s;
}
