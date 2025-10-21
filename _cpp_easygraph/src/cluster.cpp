#include "graph.h"
#include "utils.h"

inline weight_t wt(adj_dict_factory& adj, node_t u, node_t v, std::string weight, weight_t max_weight = 1) {
	auto& attr = adj[u][v];
	return (attr.count(weight) ? attr[weight] : 1) / max_weight;
}

py::list _weighted_triangles_and_degree(Graph& G,
                                        py::object nodes = py::none(),
                                        py::object weight = py::none()) {
    // ---- 0) 权重键 & 最大权重 ----
    std::string weight_key;
    bool use_weight = !weight.is_none();
    if (use_weight) {
        weight_key = weight_to_string(weight);
    }

    // 统计是否有边
    bool has_edges = false;
    for (auto it = G.adj.begin(); it != G.adj.end() && !has_edges; ++it) {
        if (!it->second.empty()) has_edges = true;
    }

    weight_t max_weight = 1;
    if (use_weight && has_edges) {
        bool assigned = false;
        for (auto uit = G.adj.begin(); uit != G.adj.end(); ++uit) {
            auto &nbrs = uit->second;
            for (auto vit = nbrs.begin(); vit != nbrs.end(); ++vit) {
                const auto &d = vit->second;  // 属性 map<string, weight_t>
                weight_t cur = 1;
                auto dit = d.find(weight_key);
                if (dit != d.end()) cur = dit->second;
                if (assigned) {
                    max_weight = std::max(max_weight, cur);
                } else {
                    assigned   = true;
                    max_weight = cur;
                }
            }
        }
    }

    // ---- 1) 确定要遍历的节点列表（Python 对象）----
    std::vector<py::object> nodes_vec;
    if (nodes.is_none()) {
        // 遍历 G.node（以 id 为键），通过 id_to_node 还原 Python 节点对象
        nodes_vec.reserve(G.node.size());
        for (auto it = G.node.begin(); it != G.node.end(); ++it) {
            node_t nid = it->first;
            py::object node_obj = G.id_to_node.attr("__getitem__")(py::cast(nid));
            nodes_vec.push_back(std::move(node_obj));
        }
    } else {
        // 遍历传入的可迭代对象，过滤不存在的节点
        for (py::handle h : nodes) {
            py::object n = py::reinterpret_borrow<py::object>(h);
            if (G.node_to_id.contains(n)) {
                nodes_vec.push_back(std::move(n));
            }
        }
    }

    // ---- 2) 主循环：每个节点的度与加权三角形数 ----
    py::list ret;
    for (const py::object &node_obj : nodes_vec) {
        // node -> id
        node_t i_id = py::cast<node_t>(G.node_to_id[node_obj]);

        // inbrs = 邻居集合（去掉自环）
        std::unordered_set<node_t> inbrs;
        auto itAdjI = G.adj.find(i_id);
        if (itAdjI != G.adj.end()) {
            auto &nbrs = itAdjI->second;
            for (auto kv = nbrs.begin(); kv != nbrs.end(); ++kv) {
                if (kv->first != i_id) inbrs.insert(kv->first);
            }
        }

        // 统计加权三角形
        std::unordered_set<node_t> seen;
        weight_t weighted_triangles = 0;

        for (const auto &j_id : inbrs) {
            seen.insert(j_id);
            weight_t wij = wt(G.adj, i_id, j_id, weight_key, max_weight);

            // 只考虑 k 不在 seen 中（避免 j-k, k-j 双计）
            for (const auto &k_id : inbrs) {
                if (seen.count(k_id)) continue;
                // 需要 j-k 相邻
                auto itAdjJ = G.adj.find(j_id);
                if (itAdjJ == G.adj.end()) continue;
                if (!itAdjJ->second.count(k_id)) continue;

                weight_t wjk = wt(G.adj, j_id, k_id, weight_key, max_weight);
                weight_t wki = wt(G.adj, k_id, i_id, weight_key, max_weight);

                weighted_triangles += std::cbrt(static_cast<double>(wij) *
                                                static_cast<double>(wjk) *
                                                static_cast<double>(wki));
            }
        }

        // 结果条目：(python_node, degree, 2 * weighted_triangles)
        // 注意：degree = inbrs.size()
        ret.append(py::make_tuple(node_obj,
                                  static_cast<int>(inbrs.size()),
                                  2.0 * static_cast<double>(weighted_triangles)));
    }

    return ret;
}
py::list _triangles_and_degree(Graph& G, py::object nodes = py::none()) {
	auto& adj = G.adj;

	// 1) 确定要处理的节点集合（Python 对象形式）
	std::vector<py::object> nodes_vec;
	if (nodes.is_none()) {
		nodes_vec.reserve(G.node.size());
		for (const auto& kv : G.node) {
			const node_t nid = kv.first;
			nodes_vec.emplace_back(G.id_to_node.attr("__getitem__")(py::cast(nid)));
		}
	} else {
		for (py::handle h : nodes) {
			py::object obj = py::reinterpret_borrow<py::object>(h);
			if (G.node_to_id.contains(obj)) nodes_vec.emplace_back(std::move(obj));
		}
	}

	// 2) 主循环
	py::list ret;
	for (const py::object& node_obj : nodes_vec) {
		// node -> id
		const node_t v = py::cast<node_t>(G.node_to_id[node_obj]);

		// 邻居集合（去掉自环）
		std::unordered_set<node_t> vs;
		if (auto it = adj.find(v); it != adj.end()) {
			for (const auto& kv : it->second) vs.insert(kv.first);
		}
		vs.erase(v);

		// 统计邻居之间的连边数（按原逻辑：有序对计数）
		weight_t ntriangles = 0;  // 原实现是对 (w,node) 有序对计数
		for (const node_t w : vs) {
			auto itW = adj.find(w);
			if (itW == adj.end()) continue;
			const auto& nbrW = itW->second;

			for (const node_t u : vs) {
				if (u == w) continue;
				ntriangles += static_cast<weight_t>(nbrW.count(u));
			}
		}

		// (python_node, degree, ntriangles)
		// 注意：原代码未除以 2，这里保持一致
		py::object py_node = G.id_to_node.attr("__getitem__")(py::cast(v));
		ret.append(py::make_tuple(py_node, static_cast<int>(vs.size()), ntriangles));
	}

	return ret;
}
py::object clustering(py::object G,
					  py::object nodes,
					  py::object weight) {
	if (py::bool_(G.attr("is_directed")())) {
		throw py::value_error("Not implemented yet");
	}

	Graph& G_ = py::cast<Graph&>(G);  // 用 py::cast 而不是 extract

	py::list td_list = weight.is_none()
		? _triangles_and_degree(G_, nodes)
		: _weighted_triangles_and_degree(G_, nodes, weight);

	py::dict clusterc;
	const std::size_t n = py::len(td_list);
	for (std::size_t i = 0; i < n; ++i) {
		py::tuple t = td_list[i].cast<py::tuple>();
		py::object v   = t[0];
		int        d   = t[1].cast<int>();
		double     tri = t[2].cast<double>();
		double c = (tri != 0.0 && d >= 2) ? (tri / (double(d) * double(d - 1))) : 0.0;
		clusterc[v] = c;
	}

	if (!nodes.is_none() && G_.node_to_id.contains(nodes)) {
		return clusterc.attr("__getitem__")(nodes);
	}
	return clusterc;
}
