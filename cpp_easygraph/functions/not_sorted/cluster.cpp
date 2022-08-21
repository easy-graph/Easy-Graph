#include "cluster.h"
#include "../../classes/graph.h"
#include "../../common/utils.h"

inline weight_t wt(adj_dict_factory& adj, node_t u, node_t v, std::string weight, weight_t max_weight = 1) {
	auto& attr = adj[u][v];
	return (attr.count(weight) ? attr[weight] : 1) / max_weight;
}

py::list _weighted_triangles_and_degree(py::object G, py::object nodes, py::object weight) {
	std::string weight_key = weight_to_string(weight);
	Graph& G_ = G.cast<Graph&>();
	weight_t max_weight = 1;
	if (weight.is_none() || G.attr("number_of_edges")().is(py::cast(0))) {
		max_weight = 1;
	}
	else {
		int assigned = 0;
		for (auto& u_info : G_.adj) {
			for (auto& v_info : u_info.second) {
				auto& d = v_info.second;
				if (assigned) {
					max_weight = std::max(max_weight, d.count(weight_key) ? d[weight_key] : 1);
				}
				else {
					assigned = 1;
					max_weight = d.count(weight_key) ? d[weight_key] : 1;
				}
			}
		}
	}
	py::list nodes_list = py::list(nodes.is_none() ? G.attr("nodes") : G.attr("nbunch_iter")(nodes));
	py::list ret = py::list();
	for (int i = 0;i < py::len(nodes_list);i++) {
		node_t i_id = (G_.node_to_id[nodes_list[i]]).cast<node_t>();
		std::unordered_set<node_t> inbrs, seen;
		auto& adj = G_.adj;
		for (const auto& pair : adj[i_id]) {
			inbrs.insert(pair.first);
		}
		inbrs.erase(i_id);
		weight_t weighted_triangles = 0;
		for (const auto& j_id : inbrs) {
			seen.insert(j_id);
			weight_t wij = wt(adj, i_id, j_id, weight_key, max_weight);
			for (const auto& k_id : inbrs) {
				if (adj[j_id].count(k_id) && !seen.count(k_id)) {
					weight_t wjk = wt(adj, j_id, k_id, weight_key, max_weight);
					weight_t wki = wt(adj, k_id, i_id, weight_key, max_weight);
					weighted_triangles += std::cbrt(wij * wjk * wki);
				}
			}
		}
		ret.append(py::make_tuple(G_.id_to_node[py::cast(i_id)], inbrs.size(), 2 * weighted_triangles));
	}
	return ret;
}

py::list _triangles_and_degree(py::object G, py::object nodes = py::none()) {
	Graph& G_ = G.cast<Graph&>();
	auto& adj = G_.adj;
	py::list nodes_list = py::list(nodes.is_none() ? G.attr("nodes") : G.attr("nbunch_iter")(nodes));
	py::list ret = py::list();
	for (int i = 0;i < py::len(nodes_list);i++) {
		node_t v = (G_.node_to_id[nodes_list[i]]).cast<node_t>();
		std::unordered_set<node_t> vs;
		for (const auto& pair : adj[v]) {
			vs.insert(pair.first);
		}
		vs.erase(v);
		weight_t ntriangles = 0;
		for (const auto& w : vs) {
			for (const auto& node : vs) {
				ntriangles += node != w && adj[w].count(node);
			}
		}
		ret.append(py::make_tuple(G_.id_to_node[py::cast(v)], vs.size(), ntriangles));
	}
	return ret;
}

py::object clustering(py::object G, py::object nodes, py::object weight) {
	py::dict clusterc = py::dict();
	if (G.attr("is_directed")().cast<bool>()) {
		PyErr_Format(PyExc_RuntimeError, "Not implemented yet");
		return py::none();
	}
	else {
		py::list td_list;
		if (!weight.is_none()) {
			td_list = _weighted_triangles_and_degree(G, nodes, weight);
		}
		else {
			td_list = _triangles_and_degree(G, nodes);
		}
		for (int i = 0;i < py::len(td_list);i++) {
			py::tuple tuple = td_list[i].cast<py::tuple>();
			py::object v = tuple[0];
			int d = tuple[1].cast<int>();
			weight_t t = tuple[2].cast<weight_t>();
			if (t == 0) {
				clusterc[v] = 0;
			}
			else {
				clusterc[v] = t / (d * (d - 1));
			}
		}
	}
	if (G.contains(nodes)) {
		return clusterc[nodes];
	}
	return clusterc;
}