#include "utils.h"
#include "../classes/graph.h"
#include "../classes/linkgraph.h"

py::object attr_to_dict(const node_attr_dict_factory& attr) {
	py::dict attr_dict = py::dict();
	for (const auto& kv : attr) {
		attr_dict[py::cast(kv.first)] = kv.second;
	}
	return attr_dict;
}

std::string weight_to_string(py::object weight) {
	py::object warn = py::module_::import("warnings").attr("warn");
	if (!py::isinstance<py::str>(weight)) {
		if (!weight.is_none()) {
			warn(py::str(weight) + py::str(" would be transformed into an instance of str."));
		}
		weight = py::str(weight);
	}
	std::string weight_key = weight.cast<std::string>();
	return weight_key;
}

py::object py_sum(py::object o) {
	py::object sum = py::module_::import("builtins").attr("sum");
	return sum(o);
}

Graph_L graph_to_linkgraph(Graph &G, bool if_directed, std::string weight_key, bool is_deg, bool is_reverse){
    int node_num = G.node.size();
    const std::vector<graph_edge>& edges = G._get_edges(if_directed);
    int edges_num = edges.size();
    Graph_L G_l(node_num, if_directed, is_deg);
	for(register int i = 0; i < edges_num; i++){
			graph_edge e = edges[i];
			edge_attr_dict_factory& edge_attr = e.attr;
			weight_t edge_weight = edge_attr.find(weight_key) != edge_attr.end() ? edge_attr[weight_key] : 1;
			if(is_reverse){
				std::swap(e.u, e.v);
			}
			G_l.add_weighted_edge(e.u, e.v, edge_weight);
			if (!if_directed){
				G_l.add_weighted_edge(e.v, e.u, edge_weight);
			}  
	}
    return G_l;
}