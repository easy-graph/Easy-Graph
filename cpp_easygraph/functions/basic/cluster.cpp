#include "cluster.h"
#include "../../classes/graph.h"
#include "../../classes/directed_graph.h"
#include "../../common/utils.h"

inline weight_t wt(adj_dict_factory& adj, node_t u, node_t v, std::string weight, weight_t max_weight = 1) {
	auto& attr = adj[u][v];
	return (attr.count(weight) ? attr[weight] : 1) / max_weight;
}

py::list _weighted_triangles_and_degree(py::object G, py::object nodes, py::object weight) {
	std::string weight_key = weight_to_string(weight);
	Graph& G_ = G.cast<Graph&>();
	auto& adj = G_.adj;
	weight_t max_weight = 1;
	if (weight.is_none() || G.attr("number_of_edges")().equal(py::cast(0))) {
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
	py::list nodes_list = py::list(G.attr("nbunch_iter")(nodes));
	py::list ret = py::list();
	for (int i = 0;i < py::len(nodes_list);i++) {
		node_t i_id = (G_.node_to_id[nodes_list[i]]).cast<node_t>();
		std::unordered_set<node_t> inbrs, seen;
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

py::list _directed_weighted_triangles_and_degree(py::object G, py::object nodes, py::object weight) {
	std::string weight_key = weight_to_string(weight);
	DiGraph& G_ = G.cast<DiGraph&>();
	auto& adj = G_.adj;
	weight_t max_weight = 1;
	if (weight.is_none() || G.attr("number_of_edges")().equal(py::cast(0))) {
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
	py::list nodes_list = py::list(G.attr("nbunch_iter")(nodes));
	py::list ret = py::list();
	for (int i = 0;i < py::len(nodes_list);i++) {
		node_t i_id = (G_.node_to_id[nodes_list[i]]).cast<node_t>();
		std::unordered_set<node_t> ipreds, isuccs;
		for (const auto& pair : G_.pred[i_id]) {
			ipreds.insert(pair.first);
		}
		ipreds.erase(i_id);
		for (const auto& pair : G_.adj[i_id]) {
			isuccs.insert(pair.first);
		}
		isuccs.erase(i_id);

		weight_t directed_triangles = 0;
		for (const auto& j_id : ipreds) {
			for (const auto& k_pair : G_.pred[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) {
					continue;
				}// jpreds
				if (ipreds.count(k_id)) { // ipreds & jpreds
					directed_triangles += std::cbrt(wt(adj, j_id, i_id, weight_key, max_weight) * wt(adj, k_id, i_id, weight_key, max_weight) * wt(adj, k_id, j_id, weight_key, max_weight));
				}
				if (isuccs.count(k_id)) { // isuccs & jpreds
					directed_triangles += std::cbrt(wt(adj, j_id, i_id, weight_key, max_weight) * wt(adj, i_id, k_id, weight_key, max_weight) * wt(adj, k_id, j_id, weight_key, max_weight));
				}
			}
			for (const auto& k_pair : G_.adj[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) {
					continue;
				}// jsuccs
				if (ipreds.count(k_id)) { // ipreds & jsuccs
					directed_triangles += std::cbrt(wt(adj, j_id, i_id, weight_key, max_weight) * wt(adj, k_id, i_id, weight_key, max_weight) * wt(adj, j_id, k_id, weight_key, max_weight));
				}
				if (isuccs.count(k_id)) { // isuccs & jsuccs
					directed_triangles += std::cbrt(wt(adj, j_id, i_id, weight_key, max_weight) * wt(adj, i_id, k_id, weight_key, max_weight) * wt(adj, j_id, k_id, weight_key, max_weight));
				}
			}
		}
		for (const auto& j_id : isuccs) {
			for (const auto& k_pair : G_.pred[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) {
					continue;
				}// jpreds
				if (ipreds.count(k_id)) { // ipreds & jpreds
					directed_triangles += std::cbrt(wt(adj, i_id, j_id, weight_key, max_weight) * wt(adj, k_id, i_id, weight_key, max_weight) * wt(adj, k_id, j_id, weight_key, max_weight));
				}
				if (isuccs.count(k_id)) { // isuccs & jpreds
					directed_triangles += std::cbrt(wt(adj, i_id, j_id, weight_key, max_weight) * wt(adj, i_id, k_id, weight_key, max_weight) * wt(adj, k_id, j_id, weight_key, max_weight));
				}
			}
			for (const auto& k_pair : G_.adj[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) {
					continue;
				}// jsuccs
				if (ipreds.count(k_id)) { // ipreds & jsuccs
					directed_triangles += std::cbrt(wt(adj, i_id, j_id, weight_key, max_weight) * wt(adj, k_id, i_id, weight_key, max_weight) * wt(adj, j_id, k_id, weight_key, max_weight));
				}
				if (isuccs.count(k_id)) { // isuccs & jsuccs
					directed_triangles += std::cbrt(wt(adj, i_id, j_id, weight_key, max_weight) * wt(adj, i_id, k_id, weight_key, max_weight) * wt(adj, j_id, k_id, weight_key, max_weight));
				}
			}
		}

		int dtotal = ipreds.size() + isuccs.size();
		int dbidirectional = 0;
		for (const auto& node : ipreds) {
			dbidirectional += isuccs.count(node);
		}
		ret.append(py::make_tuple(nodes_list[i], dtotal, dbidirectional, directed_triangles));
	}
	return ret;
}

py::list _triangles_and_degree(py::object G, py::object nodes = py::none()) {
	Graph& G_ = G.cast<Graph&>();
	auto& adj = G_.adj;
	py::list nodes_list = py::list(G.attr("nbunch_iter")(nodes));
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

py::list _directed_triangles_and_degree(py::object G, py::object nodes = py::none()) {
	DiGraph& G_ = G.cast<DiGraph&>();
	auto& adj = G_.adj;
	py::list nodes_list = py::list(G.attr("nbunch_iter")(nodes));
	py::list ret = py::list();
	for (int i = 0;i < py::len(nodes_list);i++) {
		node_t i_id = (G_.node_to_id[nodes_list[i]]).cast<node_t>();
		std::unordered_set<node_t> ipreds, isuccs;
		for (const auto& pair : G_.pred[i_id]) {
			ipreds.insert(pair.first);
		}
		ipreds.erase(i_id);
		for (const auto& pair : G_.adj[i_id]) {
			isuccs.insert(pair.first);
		}
		isuccs.erase(i_id);

		weight_t directed_triangles = 0;
		for (const auto& j_id : ipreds) {
			for (const auto& k_pair : G_.pred[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) {
					continue;
				}// jpreds
				directed_triangles += ipreds.count(k_id); // ipreds & jpreds
				directed_triangles += isuccs.count(k_id); // isuccs & jpreds
			}
			for (const auto& k_pair : G_.adj[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) {
					continue;
				}// jsuccs
				directed_triangles += ipreds.count(k_id); // ipreds & jsuccs
				directed_triangles += isuccs.count(k_id); // isuccs & jsuccs
			}
		}
		for (const auto& j_id : isuccs) {
			for (const auto& k_pair : G_.pred[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) {
					continue;
				}// jpreds
				directed_triangles += ipreds.count(k_id); // ipreds & jpreds
				directed_triangles += isuccs.count(k_id); // isuccs & jpreds
			}
			for (const auto& k_pair : G_.adj[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) {
					continue;
				}// jsuccs
				directed_triangles += ipreds.count(k_id); // ipreds & jsuccs
				directed_triangles += isuccs.count(k_id); // isuccs & jsuccs
			}
		}

		int dtotal = ipreds.size() + isuccs.size();
		int dbidirectional = 0;
		for (const auto& node : ipreds) {
			dbidirectional += isuccs.count(node);
		}
		ret.append(py::make_tuple(nodes_list[i], dtotal, dbidirectional, directed_triangles));
	}
	return ret;
}

py::object clustering(py::object G, py::object nodes, py::object weight) {
	py::dict clusterc = py::dict();
	if (G.attr("is_directed")().cast<bool>()) {
		py::list td_list;
		if (!weight.is_none()) {
			td_list = _directed_weighted_triangles_and_degree(G, nodes, weight);
		}
		else {
			td_list = _directed_triangles_and_degree(G, nodes);
		}
		for (int i = 0;i < py::len(td_list);i++) {
			py::tuple tuple = td_list[i].cast<py::tuple>();
			py::object v = tuple[0];
			int dt = tuple[1].cast<int>(), db = tuple[2].cast<int>();
			weight_t t = tuple[3].cast<weight_t>();
			if (t == 0) {
				clusterc[v] = 0;
			}
			else {
				clusterc[v] = t / ((dt * (dt - 1) - 2 * db) * 2);
			}
		}

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