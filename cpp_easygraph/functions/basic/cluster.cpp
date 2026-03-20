#include "cluster.h"
#include "../../classes/graph.h"
#include "../../classes/directed_graph.h"
#include "../../common/utils.h"
#include <omp.h>
#include <pybind11/numpy.h>

// Helper: normalized edge weight accessor
inline weight_t wt(adj_dict_factory& adj, node_t u, node_t v, std::string weight, weight_t max_weight = 1) {
	auto& attr = adj[u][v];
	return (attr.count(weight) ? attr[weight] : 1) / max_weight;
}

// Weighted triangles + degrees for undirected graphs
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

// Weighted triangles + degrees for directed graphs
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
				if (k_id == j_id) continue;
				if (ipreds.count(k_id)) {
					directed_triangles += std::cbrt(wt(adj, j_id, i_id, weight_key, max_weight) * wt(adj, k_id, i_id, weight_key, max_weight) * wt(adj, k_id, j_id, weight_key, max_weight));
				}
				if (isuccs.count(k_id)) {
					directed_triangles += std::cbrt(wt(adj, j_id, i_id, weight_key, max_weight) * wt(adj, i_id, k_id, weight_key, max_weight) * wt(adj, k_id, j_id, weight_key, max_weight));
				}
			}
			for (const auto& k_pair : G_.adj[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) continue;
				if (ipreds.count(k_id)) {
					directed_triangles += std::cbrt(wt(adj, j_id, i_id, weight_key, max_weight) * wt(adj, k_id, i_id, weight_key, max_weight) * wt(adj, j_id, k_id, weight_key, max_weight));
				}
				if (isuccs.count(k_id)) {
					directed_triangles += std::cbrt(wt(adj, j_id, i_id, weight_key, max_weight) * wt(adj, i_id, k_id, weight_key, max_weight) * wt(adj, j_id, k_id, weight_key, max_weight));
				}
			}
		}
		for (const auto& j_id : isuccs) {
			for (const auto& k_pair : G_.pred[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) continue;
				if (ipreds.count(k_id)) {
					directed_triangles += std::cbrt(wt(adj, i_id, j_id, weight_key, max_weight) * wt(adj, k_id, i_id, weight_key, max_weight) * wt(adj, k_id, j_id, weight_key, max_weight));
				}
				if (isuccs.count(k_id)) {
					directed_triangles += std::cbrt(wt(adj, i_id, j_id, weight_key, max_weight) * wt(adj, i_id, k_id, weight_key, max_weight) * wt(adj, k_id, j_id, weight_key, max_weight));
				}
			}
			for (const auto& k_pair : G_.adj[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) continue;
				if (ipreds.count(k_id)) {
					directed_triangles += std::cbrt(wt(adj, i_id, j_id, weight_key, max_weight) * wt(adj, k_id, i_id, weight_key, max_weight) * wt(adj, j_id, k_id, weight_key, max_weight));
				}
				if (isuccs.count(k_id)) {
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

// Arrays for triangle/degree results
struct TADArrays {
	std::shared_ptr<CSRGraph> csr;
	std::vector<int>          degrees;
	std::vector<int>          tri_counts;
};

struct WTADArrays {
	std::shared_ptr<CSRGraph> csr;
	std::vector<int>          degrees;
	std::vector<weight_t>     wtri_counts;
};

struct DTADArrays {
	std::vector<node_t>             nodes;
	std::unordered_map<node_t, int> node2idx;
	std::vector<int>                dtotal;
	std::vector<int>                dbidirect;
	std::vector<int>                dtri;
};

// Oriented triangle counting implementation (undirected core uses similar logic)
static TADArrays _triangles_and_degree_impl(Graph& G_) {
	auto csr = G_.gen_CSR();
	int no_of_nodes = (int)csr->nodes.size();

	if (no_of_nodes == 0) {
		return TADArrays{csr, {}, {}};
	}

	std::vector<int> degree(no_of_nodes);
#ifdef _OPENMP
	#pragma omp parallel for simd schedule(static)
#endif
	for (int i = 0; i < no_of_nodes; i++) {
		degree[i] = csr->V[i + 1] - csr->V[i];
	}

	if (!csr->oriented_valid) {
		int maxdegree = *std::max_element(degree.begin(), degree.end()) + 1;
		std::vector<int> cnt(maxdegree, 0);
		for (int i = 0; i < no_of_nodes; i++) cnt[degree[i]]++;
		for (int i = 1; i < maxdegree; i++) cnt[i] += cnt[i - 1];
		csr->order.resize(no_of_nodes);
		for (int i = no_of_nodes - 1; i >= 0; i--) csr->order[--cnt[degree[i]]] = i;

		csr->rank_arr.resize(no_of_nodes);
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (int i = 0; i < no_of_nodes; i++) {
			csr->rank_arr[csr->order[i]] = no_of_nodes - i - 1;
		}

		csr->oriented_V.assign(no_of_nodes + 1, 0);
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (int i = 0; i < no_of_nodes; i++) {
			int irank = csr->rank_arr[i];
			int out_deg = 0;
			for (int k = csr->V[i]; k < csr->V[i + 1]; k++) {
				if (csr->rank_arr[csr->E[k]] > irank) out_deg++;
			}
			csr->oriented_V[i + 1] = out_deg;
		}
		for (int i = 1; i <= no_of_nodes; i++) {
			csr->oriented_V[i] += csr->oriented_V[i - 1];
		}

		csr->oriented_E.resize(csr->oriented_V[no_of_nodes]);
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (int i = 0; i < no_of_nodes; i++) {
			int irank = csr->rank_arr[i];
			int ptr   = csr->oriented_V[i];
			for (int k = csr->V[i]; k < csr->V[i + 1]; k++) {
				int j = csr->E[k];
				if (csr->rank_arr[j] > irank)
					csr->oriented_E[ptr++] = j;
			}
		}

		csr->oriented_valid = true;
	}

	const std::vector<int>& oriented_V = csr->oriented_V;
	const std::vector<int>& oriented_E = csr->oriented_E;

	struct TLBuf {
		std::vector<int> tc;
		std::vector<int> marker;
		std::vector<int> dirty;
		int cap = 0;
		void ensure(int n) {
			if (n > cap) {
				tc.resize(n, 0);
				marker.resize(n, 0);
				cap = n;
			}
		}
	};
	thread_local TLBuf tl_buf;

	std::vector<int> tri_count(no_of_nodes, 0);

#ifdef _OPENMP
	#pragma omp parallel
#endif
	{
		tl_buf.ensure(no_of_nodes);
		int* my_tc     = tl_buf.tc.data();
		int* my_marker = tl_buf.marker.data();

#ifdef _OPENMP
		#pragma omp for schedule(guided)
#endif
		for (int node = 0; node < no_of_nodes; node++) {
			for (int k = oriented_V[node]; k < oriented_V[node + 1]; k++)
				my_marker[oriented_E[k]] = node + 1;

			for (int k = oriented_V[node]; k < oriented_V[node + 1]; k++) {
				int nei = oriented_E[k];
				for (int k2 = oriented_V[nei]; k2 < oriented_V[nei + 1]; k2++) {
					int nei2 = oriented_E[k2];
					if (my_marker[nei2] == node + 1) {
						if (!my_tc[node])  tl_buf.dirty.push_back(node);
						my_tc[node]++;
						if (!my_tc[nei])   tl_buf.dirty.push_back(nei);
						my_tc[nei]++;
						if (!my_tc[nei2])  tl_buf.dirty.push_back(nei2);
						my_tc[nei2]++;
					}
				}
			}
		}

#ifdef _OPENMP
		for (int x : tl_buf.dirty) {
			#pragma omp atomic
			tri_count[x] += my_tc[x];
			my_tc[x] = 0;
		}
#else
		for (int x : tl_buf.dirty) {
			tri_count[x] += my_tc[x];
			my_tc[x] = 0;
		}
#endif
		tl_buf.dirty.clear();
	}

	return TADArrays{csr, std::move(degree), std::move(tri_count)};
}

// Weighted triangles implementation (undirected)
static WTADArrays _weighted_triangles_and_degree_impl(Graph& G_, const std::string& weight_key) {
	auto csr = G_.gen_CSR();
	int N = (int)csr->nodes.size();
	if (N == 0) {
		return WTADArrays{csr, {}, {}};
	}

	weight_t max_weight = 1.0f;
	{
		int assigned = 0;
		for (auto& u_info : G_.adj) {
			for (auto& v_info : u_info.second) {
				auto& d = v_info.second;
				weight_t w = d.count(weight_key) ? d.at(weight_key) : 1.0f;
				if (!assigned) { max_weight = w; assigned = 1; }
				else max_weight = std::max(max_weight, w);
			}
		}
	}
	if (max_weight == 0.0f) max_weight = 1.0f;

	int M = (int)csr->E.size();
	std::vector<weight_t> W(M);
	auto& adj = G_.adj;
	for (int i = 0; i < N; i++) {
		node_t ni = csr->nodes[i];
		for (int k = csr->V[i]; k < csr->V[i + 1]; k++) {
			node_t nj = csr->nodes[csr->E[k]];
			auto& attr = adj[ni][nj];
			W[k] = (attr.count(weight_key) ? attr.at(weight_key) : 1.0f) / max_weight;
		}
	}

	std::vector<int> degree(N);
#ifdef _OPENMP
	#pragma omp parallel for simd schedule(static)
#endif
	for (int i = 0; i < N; i++) {
		degree[i] = csr->V[i + 1] - csr->V[i];
	}

	struct TLWBuf {
		std::vector<weight_t> mark;
		std::vector<int>      dirty;
		int cap = 0;
		void ensure(int n) {
			if (n > cap) {
				mark.assign(n, 0.0f);
				cap = n;
			}
		}
	};
	thread_local TLWBuf tl_wbuf;

	std::vector<weight_t> wtri(N, 0.0f);

#ifdef _OPENMP
	#pragma omp parallel
#endif
	{
		tl_wbuf.ensure(N);
		weight_t* my_mark = tl_wbuf.mark.data();

#ifdef _OPENMP
		#pragma omp for schedule(dynamic, 64)
#endif
		for (int i = 0; i < N; i++) {
			for (int k = csr->V[i]; k < csr->V[i + 1]; k++) {
				int j = csr->E[k];
				my_mark[j] = W[k];
				tl_wbuf.dirty.push_back(j);
			}

			weight_t tri_i = 0.0f;
			for (int k = csr->V[i]; k < csr->V[i + 1]; k++) {
				int j = csr->E[k];
				weight_t wij = W[k];
				for (int k2 = csr->V[j]; k2 < csr->V[j + 1]; k2++) {
					int kk = csr->E[k2];
					weight_t mk = my_mark[kk];
					if (mk > 0.0f && kk != j)
						tri_i += std::cbrt(wij * W[k2] * mk);
				}
			}
			wtri[i] = tri_i * 0.5f;

			for (int x : tl_wbuf.dirty) my_mark[x] = 0.0f;
			tl_wbuf.dirty.clear();
		}
	}

	return WTADArrays{csr, std::move(degree), std::move(wtri)};
}

// Directed triangles + degree using forward/reverse CSR
static DTADArrays _directed_triangles_and_degree_impl(DiGraph& G_) {
	std::unordered_map<node_t, int> node2idx;
	std::vector<node_t> nodes;
	for (auto& u : G_.adj) {
		if (!node2idx.count(u.first)) {
			node2idx[u.first] = (int)nodes.size();
			nodes.push_back(u.first);
		}
	}
	for (auto& u : G_.pred) {
		if (!node2idx.count(u.first)) {
			node2idx[u.first] = (int)nodes.size();
			nodes.push_back(u.first);
		}
	}
	std::sort(nodes.begin(), nodes.end());
	for (int i = 0; i < (int)nodes.size(); i++) node2idx[nodes[i]] = i;
	int N = (int)nodes.size();
	if (N == 0) return DTADArrays{{}, {}, {}, {}, {}};

	std::vector<int> fwd_V(N + 1, 0);
	for (int i = 0; i < N; i++) {
		auto it = G_.adj.find(nodes[i]);
		if (it == G_.adj.end()) continue;
		for (auto& v : it->second) {
			auto jt = node2idx.find(v.first);
			if (jt != node2idx.end() && jt->second != i) fwd_V[i + 1]++;
		}
	}
	for (int i = 0; i < N; i++) fwd_V[i + 1] += fwd_V[i];
	std::vector<int> fwd_E(fwd_V[N]);
	for (int i = 0; i < N; i++) {
		auto it = G_.adj.find(nodes[i]);
		if (it == G_.adj.end()) continue;
		int ptr = fwd_V[i];
		for (auto& v : it->second) {
			auto jt = node2idx.find(v.first);
			if (jt != node2idx.end() && jt->second != i) fwd_E[ptr++] = jt->second;
		}
	}

	std::vector<int> rev_V(N + 1, 0);
	for (int i = 0; i < N; i++) {
		auto it = G_.pred.find(nodes[i]);
		if (it == G_.pred.end()) continue;
		for (auto& v : it->second) {
			auto jt = node2idx.find(v.first);
			if (jt != node2idx.end() && jt->second != i) rev_V[i + 1]++;
		}
	}
	for (int i = 0; i < N; i++) rev_V[i + 1] += rev_V[i];
	std::vector<int> rev_E(rev_V[N]);
	for (int i = 0; i < N; i++) {
		auto it = G_.pred.find(nodes[i]);
		if (it == G_.pred.end()) continue;
		int ptr = rev_V[i];
		for (auto& v : it->second) {
			auto jt = node2idx.find(v.first);
			if (jt != node2idx.end() && jt->second != i) rev_E[ptr++] = jt->second;
		}
	}

	std::vector<int> dtotal(N);
#ifdef _OPENMP
	#pragma omp parallel for simd schedule(static)
#endif
	for (int i = 0; i < N; i++) {
		dtotal[i] = (fwd_V[i + 1] - fwd_V[i]) + (rev_V[i + 1] - rev_V[i]);
	}

	struct TLDBuf {
		std::vector<int8_t> mark;
		std::vector<int>    dirty;
		int cap = 0;
		void ensure(int n) {
			if (n > cap) { mark.assign(n, 0); cap = n; }
		}
	};
	thread_local TLDBuf tl_dbuf;

	std::vector<int> dtri(N, 0);
	std::vector<int> dbidirect(N, 0);

#ifdef _OPENMP
	#pragma omp parallel
#endif
	{
		tl_dbuf.ensure(N);
		int8_t* my_mark = tl_dbuf.mark.data();

#ifdef _OPENMP
		#pragma omp for schedule(dynamic, 64)
#endif
		for (int i = 0; i < N; i++) {
			for (int k = rev_V[i]; k < rev_V[i + 1]; k++) {
				int j = rev_E[k];
				if (!my_mark[j]) tl_dbuf.dirty.push_back(j);
				my_mark[j] |= 1;
			}
			for (int k = fwd_V[i]; k < fwd_V[i + 1]; k++) {
				int j = fwd_E[k];
				if (!my_mark[j]) tl_dbuf.dirty.push_back(j);
				my_mark[j] |= 2;
			}

			int bd = 0;
			for (int x : tl_dbuf.dirty) bd += (my_mark[x] == 3);
			dbidirect[i] = bd;

			int tri_i = 0;
			for (int k = rev_V[i]; k < rev_V[i + 1]; k++) {
				int j = rev_E[k];
				for (int k2 = rev_V[j]; k2 < rev_V[j + 1]; k2++) {
					int8_t mk = my_mark[rev_E[k2]];
					tri_i += (mk & 1) != 0;
					tri_i += (mk & 2) != 0;
				}
				for (int k2 = fwd_V[j]; k2 < fwd_V[j + 1]; k2++) {
					int8_t mk = my_mark[fwd_E[k2]];
					tri_i += (mk & 1) != 0;
					tri_i += (mk & 2) != 0;
				}
			}
			for (int k = fwd_V[i]; k < fwd_V[i + 1]; k++) {
				int j = fwd_E[k];
				for (int k2 = rev_V[j]; k2 < rev_V[j + 1]; k2++) {
					int8_t mk = my_mark[rev_E[k2]];
					tri_i += (mk & 1) != 0;
					tri_i += (mk & 2) != 0;
				}
				for (int k2 = fwd_V[j]; k2 < fwd_V[j + 1]; k2++) {
					int8_t mk = my_mark[fwd_E[k2]];
					tri_i += (mk & 1) != 0;
					tri_i += (mk & 2) != 0;
				}
			}
			dtri[i] = tri_i;

			for (int x : tl_dbuf.dirty) my_mark[x] = 0;
			tl_dbuf.dirty.clear();
		}
	}

	return DTADArrays{
		std::move(nodes),
		std::move(node2idx),
		std::move(dtotal),
		std::move(dbidirect),
		std::move(dtri)
	};
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
				if (k_id == j_id) continue;
				directed_triangles += ipreds.count(k_id);
				directed_triangles += isuccs.count(k_id);
			}
			for (const auto& k_pair : G_.adj[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) continue;
				directed_triangles += ipreds.count(k_id);
				directed_triangles += isuccs.count(k_id);
			}
		}
		for (const auto& j_id : isuccs) {
			for (const auto& k_pair : G_.pred[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) continue;
				directed_triangles += ipreds.count(k_id);
				directed_triangles += isuccs.count(k_id);
			}
			for (const auto& k_pair : G_.adj[j_id]) {
				node_t k_id = k_pair.first;
				if (k_id == j_id) continue;
				directed_triangles += ipreds.count(k_id);
				directed_triangles += isuccs.count(k_id);
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
		if (!weight.is_none()) {
			py::list td_list = _directed_weighted_triangles_and_degree(G, nodes, weight);
			for (int i = 0; i < py::len(td_list); i++) {
				py::tuple tuple = td_list[i].cast<py::tuple>();
				py::object v = tuple[0];
				int dt = tuple[1].cast<int>(), db = tuple[2].cast<int>();
				weight_t t = tuple[3].cast<weight_t>();
				if (t == 0) { clusterc[v] = 0; }
				else { clusterc[v] = t / ((dt * (dt - 1) - 2 * db) * 2); }
			}
		} else {
			DiGraph& G_ = G.cast<DiGraph&>();
			DTADArrays dtad;
			{ py::gil_scoped_release _release; dtad = _directed_triangles_and_degree_impl(G_); }
			int N = (int)dtad.nodes.size();
			if (nodes.is_none()) {
				for (int idx = 0; idx < N; idx++) {
					node_t     v       = dtad.nodes[idx];
					py::object py_node = G_.id_to_node[py::cast(v)];
					int dt = dtad.dtotal[idx], db = dtad.dbidirect[idx];
					int t  = dtad.dtri[idx];
					int denom = (dt * (dt - 1) - 2 * db) * 2;
					clusterc[py_node] = (t == 0 || denom == 0)
						? py::cast(0)
						: py::cast((weight_t)t / denom);
				}
			} else {
				py::list nodes_list = py::list(G.attr("nbunch_iter")(nodes));
				int n = (int)py::len(nodes_list);
				for (int i = 0; i < n; i++) {
					py::object py_node = nodes_list[i].cast<py::object>();
					node_t     v       = G_.node_to_id[py_node].cast<node_t>();
					auto       it      = dtad.node2idx.find(v);
					if (it == dtad.node2idx.end()) continue;
					int idx = it->second;
					int dt = dtad.dtotal[idx], db = dtad.dbidirect[idx];
					int t  = dtad.dtri[idx];
					int denom = (dt * (dt - 1) - 2 * db) * 2;
					clusterc[py_node] = (t == 0 || denom == 0)
						? py::cast(0)
						: py::cast((weight_t)t / denom);
				}
			}
		}
	}
	else {
		if (!weight.is_none()) {
			std::string wkey = weight_to_string(weight);
			Graph& G_ = G.cast<Graph&>();
			WTADArrays wtad;
			{ py::gil_scoped_release _release; wtad = _weighted_triangles_and_degree_impl(G_, wkey); }
			int N = (int)wtad.csr->nodes.size();
			if (nodes.is_none()) {
				for (int idx = 0; idx < N; idx++) {
					node_t     v       = wtad.csr->nodes[idx];
					py::object py_node = G_.id_to_node[py::cast(v)];
					int        d       = wtad.degrees[idx];
					weight_t   t       = 2.0f * wtad.wtri_counts[idx];
					clusterc[py_node]  = (t == 0)
						? py::cast(0)
						: py::cast(t / ((weight_t)d * (d - 1)));
				}
			} else {
				py::list nodes_list = py::list(G.attr("nbunch_iter")(nodes));
				int n = (int)py::len(nodes_list);
				for (int i = 0; i < n; i++) {
					py::object py_node = nodes_list[i].cast<py::object>();
					node_t     v       = G_.node_to_id[py_node].cast<node_t>();
					auto       it      = wtad.csr->node2idx.find(v);
					if (it == wtad.csr->node2idx.end()) continue;
					int        idx     = it->second;
					int        d       = wtad.degrees[idx];
					weight_t   t       = 2.0f * wtad.wtri_counts[idx];
					clusterc[py_node]  = (t == 0)
						? py::cast(0)
						: py::cast(t / ((weight_t)d * (d - 1)));
				}
			}
		} else {
			Graph& G_  = G.cast<Graph&>();
			TADArrays tad;
			{ py::gil_scoped_release _release; tad = _triangles_and_degree_impl(G_); }
			int    N   = (int)tad.csr->nodes.size();
			if (nodes.is_none()) {
				for (int idx = 0; idx < N; idx++) {
					node_t     v       = tad.csr->nodes[idx];
					py::object py_node = G_.id_to_node[py::cast(v)];
					int        d       = tad.degrees[idx];
					int        t       = 2 * tad.tri_counts[idx];
					clusterc[py_node]  = (t == 0)
						? py::cast(0)
						: py::cast((weight_t)t / ((weight_t)d * (d - 1)));
				}
			} else {
				py::list nodes_list = py::list(G.attr("nbunch_iter")(nodes));
				int n = (int)py::len(nodes_list);
				for (int i = 0; i < n; i++) {
					py::object py_node = nodes_list[i].cast<py::object>();
					node_t     v       = G_.node_to_id[py_node].cast<node_t>();
					auto       it      = tad.csr->node2idx.find(v);
					if (it == tad.csr->node2idx.end()) continue;
					int        idx     = it->second;
					int        d       = tad.degrees[idx];
					int        t       = 2 * tad.tri_counts[idx];
					clusterc[py_node]  = (t == 0)
						? py::cast(0)
						: py::cast((weight_t)t / ((weight_t)d * (d - 1)));
				}
			}
		}
	}
	if (G.contains(nodes)) {
		return clusterc[nodes];
	}
	return clusterc;
}

// Average clustering (returns double)
double cpp_average_clustering(py::object G) {
	if (G.attr("is_directed")().cast<bool>()) {
		py::dict d = clustering(G, py::none(), py::none()).cast<py::dict>();
		double sum = 0.0;
		int cnt = 0;
		for (auto item : d) { sum += item.second.cast<double>(); cnt++; }
		return cnt > 0 ? sum / cnt : 0.0;
	}

	Graph& G_ = G.cast<Graph&>();
	TADArrays tad;
	{ py::gil_scoped_release _release; tad = _triangles_and_degree_impl(G_); }

	int N = (int)tad.csr->nodes.size();
	double sum = 0.0;
	int valid = 0;
	for (int i = 0; i < N; i++) {
		int d = tad.degrees[i];
		if (d < 2) continue;
		int t = 2 * tad.tri_counts[i];
		sum += (double)t / ((double)d * (d - 1));
		valid++;
	}
	return valid > 0 ? sum / valid : 0.0;
}

// Return node ids and coefficients arrays for undirected graphs
py::tuple cpp_clustering_array(py::object G) {
	if (G.attr("is_directed")().cast<bool>()) {
		throw std::runtime_error("cpp_clustering_array only supports undirected graphs");
	}

	Graph& G_ = G.cast<Graph&>();
	TADArrays tad;
	{ py::gil_scoped_release _release; tad = _triangles_and_degree_impl(G_); }

	int N = (int)tad.csr->nodes.size();

	py::array_t<int32_t> node_ids(N);
	py::array_t<double>  coeffs(N);
	auto id_buf = node_ids.mutable_unchecked<1>();
	auto c_buf  = coeffs.mutable_unchecked<1>();

	for (int i = 0; i < N; i++) {
		id_buf(i) = (int32_t)tad.csr->nodes[i];
		int d = tad.degrees[i];
		int t = 2 * tad.tri_counts[i];
		c_buf(i) = (d < 2 || t == 0)
			? 0.0
			: (double)t / ((double)d * (d - 1));
	}
	return py::make_tuple(node_ids, coeffs);
}