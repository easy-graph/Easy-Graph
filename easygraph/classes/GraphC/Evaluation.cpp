#include "Evaluation.h"
#include "Utils.h"

struct pair_hash
{
	template<class T1, class T2>
	std::size_t operator() (const std::pair<T1, T2>& p) const
	{
		auto h1 = std::hash<T1>()(p.first);
		auto h2 = std::hash<T2>()(p.second);
		return h1 ^ h2;
	}
};

std::unordered_map<std::pair<Graph::node_t, Graph::node_t>, Graph::weight_t, pair_hash> sum_nmw_rec, max_nmw_rec, local_constraint_rec;

enum norm_t {
	sum, max
};


Graph::weight_t mutual_weight(Graph::adj_dict_factory& G, Graph::node_t u, Graph::node_t v, std::string weight) {
	Graph::weight_t a_uv = 0, a_vu = 0;
	if (G.count(u) && G[u].count(v)) {
		Graph::edge_attr_dict_factory& guv = G[u][v];
		a_uv = guv.count(weight) ? guv[weight] : 1;
	}
	if (G.count(v) && G[v].count(u)) {
		Graph::edge_attr_dict_factory& gvu = G[v][u];
		a_uv = gvu.count(weight) ? gvu[weight] : 1;
	}
	return a_uv + a_vu;
}

Graph::weight_t  normalized_mutual_weight(Graph::adj_dict_factory& G, Graph::node_t u, Graph::node_t v, std::string weight, norm_t norm = sum) {
	std::pair<Graph::node_t, Graph::node_t> edge = std::make_pair(u, v);
	auto& nmw_rec = (norm == sum) ? sum_nmw_rec : max_nmw_rec;
	if (nmw_rec.count(edge)) {
		return nmw_rec[edge];
	}
	else {
		Graph::weight_t scale = 0;
		for (auto& w : G[u]) {
			Graph::weight_t temp_weight = mutual_weight(G, u, w.first, weight);
			scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
		}
		Graph::weight_t nmw = scale ? (mutual_weight(G, u, v, weight) / scale) : 0;
		nmw_rec[edge] = nmw;
		return nmw;
	}
}

Graph::weight_t local_constraint(Graph::adj_dict_factory& G, Graph::node_t u, Graph::node_t v, std::string weight = "None") {
	std::pair<Graph::node_t, Graph::node_t> edge = std::make_pair(u, v);
	if (local_constraint_rec.count(edge)) {
		return local_constraint_rec[edge];
	}
	else {
		Graph::weight_t direct = normalized_mutual_weight(G, u, v, weight);
		Graph::weight_t indirect = 0;
		for (auto& w : G[u]) {
			indirect += normalized_mutual_weight(G, u, w.first, weight) * normalized_mutual_weight(G, w.first, v, weight);
		}
		Graph::weight_t result = pow((direct + indirect), 2);
		local_constraint_rec[edge] = result;
		return result;
	}
}

std::pair<Graph::node_t, Graph::weight_t> compute_constraint_of_v(Graph::adj_dict_factory& G, Graph::node_t v, std::string weight) {
	Graph::weight_t constraint_of_v = 0;
	if (G[v].size() == 0) {
		constraint_of_v = Py_NAN;
	}
	else {
		for (const auto& n : G[v]) {
			constraint_of_v += local_constraint(G, v, n.first, weight);
		}
	}
	return std::make_pair(v, constraint_of_v);
}

py::object constraint(py::object G, py::object nodes, py::object weight, py::object n_workers) {
	std::string weight_key = weight_to_string(weight);
	sum_nmw_rec.clear();
	max_nmw_rec.clear();
	local_constraint_rec.clear();
	if (nodes == py::object()) {
		nodes = G.attr("nodes");
	}
	py::list nodes_list = py::list(nodes);
	py::list constraint_results = py::list();
	Graph& G_ = py::extract<Graph&>(G);
	for (int i = 0;i < py::len(nodes_list);i++) {
		py::object v = nodes_list[i];
		Graph::node_t v_id = py::extract<Graph::node_t>(G_.node_to_id[v]);
		std::pair<Graph::node_t, Graph::weight_t> constraint_pair = compute_constraint_of_v(G_.adj, v_id, weight_key);
		py::tuple constraint_of_v = py::make_tuple(G_.id_to_node[constraint_pair.first], constraint_pair.second);
		constraint_results.append(constraint_of_v);
	}
	py::dict constraint = py::dict(constraint_results);
	return constraint;
}

Graph::weight_t redundancy(Graph::adj_dict_factory& G, Graph::node_t u, Graph::node_t v, std::string weight = "None") {
	Graph::weight_t r = 0;
	for (const auto& neighbor_info : G[u]) {
		Graph::node_t w = neighbor_info.first;
		r += normalized_mutual_weight(G, u, w, weight) * normalized_mutual_weight(G, v, w, weight, max);
	}
	return 1 - r;
}

py::object effective_size(py::object G, py::object nodes, py::object weight, py::object n_workers) {
	Graph& G_ = py::extract<Graph&>(G);
	sum_nmw_rec.clear();
	max_nmw_rec.clear();
	py::dict effective_size = py::dict();
	if (nodes == py::object()) {
		nodes = G;
	}
	nodes = py::list(nodes);
	if (!G.attr("is_directed")() && weight == py::object()) {
		for (int i = 0;i < py::len(nodes);i++) {
			py::object v = nodes[i];
			if (py::len(G[v]) == 0) {
				effective_size[v] = py::object(Py_NAN);
				continue;
			}
			py::object E = G.attr("ego_subgraph")(v);
			if (py::len(E) > 1) {
				Graph::weight_t size = py::extract<Graph::weight_t>(E.attr("size")());
				effective_size[v] = py::len(E) - 1 - (2.0 * size) / (py::len(E) - 1);
			}
			else {
				effective_size[v] = 0;
			}
		}
	}
	else {
		std::string weight_key = weight_to_string(weight);
		for (int i = 0;i < py::len(nodes);i++) {
			py::object v = nodes[i];
			if (py::len(G[v]) == 0) {
				effective_size[v] = py::object(Py_NAN);
				continue;
			}
			Graph::weight_t redundancy_sum = 0;
			Graph::node_t v_id = py::extract<Graph::node_t>(G_.node_to_id[v]);
			for (const auto& neighbor_info : G_.adj[v_id]) {
				Graph::node_t u_id = neighbor_info.first;
				redundancy_sum += redundancy(G_.adj, v_id, u_id, weight_key);
			}
			effective_size[v] = redundancy_sum;
		}
	}
	return effective_size;
}

py::object hierarchy(py::object G, py::object nodes, py::object weight, py::object n_workers) {
	sum_nmw_rec.clear();
	max_nmw_rec.clear();
	local_constraint_rec.clear();
	std::string weight_key = weight_to_string(weight);
	if (nodes == py::object()) {
		nodes = G.attr("nodes");
	}
	py::list nodes_list = py::list(nodes);

	Graph& G_ = py::extract<Graph&>(G);
	py::dict hierarchy = py::dict();

	for (int i = 0;i < py::len(nodes_list);i++) {
		py::object v = nodes_list[i];
		py::object E = G.attr("ego_subgraph")(v);

		int n = py::len(E) - 1;

		Graph::weight_t C = 0;
		std::map<Graph::node_t, Graph::weight_t> c;
		py::list neighbors_of_v = py::list(G.attr("neighbors")(v));

		for (int j = 0;j < py::len(neighbors_of_v);j++) {
			py::object w = neighbors_of_v[j];
			Graph::node_t v_id = py::extract<Graph::node_t>(G_.node_to_id[v]);
			Graph::node_t w_id = py::extract<Graph::node_t>(G_.node_to_id[w]);
			C += local_constraint(G_.adj, v_id, w_id, weight_key);
			c[w_id] = local_constraint(G_.adj, v_id, w_id, weight_key);
		}
		if (n > 1) {
			Graph::weight_t hierarchy_sum = 0;
			for (int k = 0;k < py::len(neighbors_of_v);k++) {
				py::object w = neighbors_of_v[k];
				Graph::node_t w_id = py::extract<Graph::node_t>(G_.node_to_id[w]);
				hierarchy_sum += c[w_id] / C * n * log(c[w_id] / C * n) / (n * log(n));
			}
			hierarchy[v] = hierarchy_sum;
		}
		if (!hierarchy.has_key(v)) {
			hierarchy[v] = 0;
		}
	}
	return hierarchy;
}