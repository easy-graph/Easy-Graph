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