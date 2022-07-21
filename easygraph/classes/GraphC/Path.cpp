#include "Path.h"
#include "Utils.h"
#include <queue>
#include <vector>

py::object _dijkstra_multisource(py::object G, py::object sources, py::object weight, py::object target) {
	Graph& G_ = py::extract<Graph&>(G);
	std::string weight_key = weight_to_string(weight);
	Graph::node_t target_id = py::extract<Graph::node_t>(G_.node_to_id.get(target, -1));
	std::map<Graph::node_t, Graph::weight_t> dist, seen;
	std::priority_queue<std::pair<Graph::weight_t, Graph::node_t>, std::vector<std::pair<Graph::weight_t, Graph::node_t> >, std::greater<> > Q;
	py::list sources_list = py::list(sources);
	for (int i = 0;i < py::len(sources_list);i++) {
		Graph::node_t source = py::extract<Graph::node_t>(G_.node_to_id[sources_list[i]]);
		seen[source] = 0;
		Q.push(std::make_pair(0, source));
	}
	while (!Q.empty()) {
		std::pair<Graph::weight_t, Graph::node_t> node = Q.top();
		Q.pop();
		Graph::weight_t d = node.first;
		Graph::node_t v = node.second;
		if (dist.count(v)) {
			continue;
		}
		dist[v] = d;
		if (v == target_id) {
			break;
		}
		Graph::adj_dict_factory& adj = G_.adj;
		for (auto& neighbor_info : adj[v]) {
			Graph::node_t u = neighbor_info.first;
			Graph::weight_t cost = neighbor_info.second.count(weight_key) ? neighbor_info.second[weight_key] : 1;
			Graph::weight_t vu_dist = dist[v] + cost;
			if (dist.count(u)) {
				if (vu_dist < dist[u]) {
					PyErr_Format(PyExc_ValueError, "Contradictory paths found: negative weights?");
					return py::object();
				}
			}
			else if (!seen.count(u) || vu_dist < seen[u]) {
				seen[u] = vu_dist;
				Q.push(std::make_pair(vu_dist, u));
			}
			else {
				continue;
			}
		}
	}
	py::dict pydist = py::dict();
	for (const auto& kv : dist) {
		pydist[G_.id_to_node[kv.first]] = kv.second;
	}
	return pydist;
}