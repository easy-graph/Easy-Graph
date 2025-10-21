#include "path.h"
#include "graph.h"
#include "utils.h"

#include <numeric>

py::dict _dijkstra_multisource(const py::object &G,
                                      const py::object &sources,
                                      const py::object &weight,
                                      const py::object &target) {
  Graph &G_ = G.cast<Graph &>();
  const std::string weight_key = weight_to_string(weight);

  // target_id = G_.node_to_id.get(target, -1)
  const node_t target_id =
      G_.node_to_id.attr("get")(target, py::int_(-1)).cast<node_t>();

  std::map<node_t, weight_t> dist;
  std::map<node_t, weight_t> seen;

  using QItem = std::pair<weight_t, node_t>;
  std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> Q;

  // sources 列表
  py::list sources_list = py::list(sources);
  for (py::ssize_t i = 0; i < py::len(sources_list); ++i) {
    const node_t s_id = G_.node_to_id[sources_list[i]].cast<node_t>();
    seen[s_id] = weight_t(0);
    Q.emplace(weight_t(0), s_id);
  }

  // Dijkstra
  while (!Q.empty()) {
    const auto node = Q.top();
    Q.pop();
    const weight_t d = node.first;
    const node_t v = node.second;

    if (dist.count(v)) {
      continue;
    }
    dist[v] = d;

    if (v == target_id) {
      break; // 已到目标
    }

    auto &adj = G_.adj;                  // 邻接结构
    for (auto &neighbor_info : adj[v]) { // neighbor_info: (u, attr)
      const node_t u = neighbor_info.first;

      // attr: 例如 std::unordered_map<std::string, weight_t>
      auto &attr = neighbor_info.second;
      const weight_t cost =
          (attr.count(weight_key) ? attr[weight_key] : weight_t(1));

      const weight_t vu_dist = dist[v] + cost;

      if (dist.count(u)) {
        if (vu_dist < dist[u]) {
          throw py::value_error("Contradictory paths found: negative weights?");
        }
        // 否则已是最优，跳过
      } else if (!seen.count(u) || vu_dist < seen[u]) {
        seen[u] = vu_dist;
        Q.emplace(vu_dist, u);
      } else {
        // 已有更优 seen[u]
      }
    }
  }

  // 转回 Python 映射：node_obj -> distance
  py::dict pydist;
  for (const auto &kv : dist) {
    const node_t vid = kv.first;
    const weight_t d = kv.second;
  	py::object key = G_.id_to_node.attr("__getitem__")(py::cast(vid));
    pydist[key] = py::cast(d);
  }
  return pydist;
}
py::dict Prim(Graph& G) {
    py::dict result_dict;

    // 结果图（临时存：id -> (id -> weight)）
    std::unordered_map<node_t, std::unordered_map<node_t, weight_t>> res_dict;

    auto& nodes = G.node;
    auto& adj   = G.adj;

    // 收集节点，初始化 selected/candidate
    std::vector<node_t> selected;
    std::vector<node_t> candidate;
    selected.reserve(nodes.size());
    candidate.reserve(nodes.size());

    for (node_dict_factory::const_iterator it = nodes.begin(); it != nodes.end(); ++it) {
        node_t node_id = it->first;
        // 先在结果字典中为每个 python 节点放一个空 dict
        py::object node_obj = G.id_to_node.attr("__getitem__")(py::cast(node_id));
        result_dict[node_obj] = py::dict();

        if (selected.empty()) {
            selected.emplace_back(node_id);
        } else {
            candidate.emplace_back(node_id);
        }
    }

    if (selected.empty()) {
        // 空图
        return result_dict;
    }

    const weight_t INF = std::numeric_limits<weight_t>::infinity();

    // 朴素 Prim：每次在 selected 与 candidate 间找最小权边
    while (!candidate.empty()) {
        node_t best_u = static_cast<node_t>(-1);
        node_t best_v = static_cast<node_t>(-1);
        weight_t best_w = INF;

        for (std::size_t i = 0; i < selected.size(); ++i) {
            const node_t u = selected[i];

            // 找 u 的邻接
            adj_attr_dict_factory node_adj;
            if (adj.find(u) != adj.end()) {
                node_adj = adj.at(u);
            } else {
                continue;
            }

            for (std::size_t j = 0; j < candidate.size(); ++j) {
                const node_t v = candidate[j];

                weight_t edge_weight = INF;
                // 是否存在 u->v 边
                adj_attr_dict_factory::const_iterator it_uv = node_adj.find(v);
                if (it_uv != node_adj.end()) {
                    const edge_attr_dict_factory& eattr = it_uv->second;
                    edge_attr_dict_factory::const_iterator wit = eattr.find("weight");
                    edge_weight = (wit != eattr.end()) ? wit->second : static_cast<weight_t>(1);
                }

                if (nodes.find(u) != nodes.end() && edge_weight < best_w) {
                    best_u = u; best_v = v; best_w = edge_weight;
                }
            }
        }

        if (best_u != static_cast<node_t>(-1) && best_v != static_cast<node_t>(-1)) {
            // 选入最小边 (best_u, best_v)
            res_dict[best_u][best_v] = best_w;

            // v 从 candidate -> selected
            selected.emplace_back(best_v);
            std::vector<node_t>::iterator it = std::find(candidate.begin(), candidate.end(), best_v);
            if (it != candidate.end()) candidate.erase(it);
        } else {
            // 剩余节点与 selected 之间不连通（图不连通，Prim 停止）
            break;
        }
    }

    // 将 res_dict 中的 id 映射成 python 节点对象，并写回 result_dict
    for (std::unordered_map<node_t, std::unordered_map<node_t, weight_t>>::const_iterator it = res_dict.begin();
         it != res_dict.end(); ++it) {
        const node_t u = it->first;
        const std::unordered_map<node_t, weight_t>& nbrs = it->second;

        py::object u_obj = G.id_to_node.attr("__getitem__")(py::cast(u));
        py::dict u_dict  = result_dict.attr("__getitem__")(u_obj).cast<py::dict>();

        for (std::unordered_map<node_t, weight_t>::const_iterator jt = nbrs.begin();
             jt != nbrs.end(); ++jt) {
            const node_t v = jt->first;
            const weight_t w = jt->second;

            py::object v_obj = G.id_to_node.attr("__getitem__")(py::cast(v));
            u_dict[v_obj] = w;
        }
    }

    return result_dict;
}bool comp(const std::pair<std::pair<node_t, node_t>, weight_t> &a,
          const std::pair<std::pair<node_t, node_t>, weight_t> &b) {
  return a.second < b.second;
}
py::dict Kruskal(Graph& G) {
    py::dict result_dict;

    auto& nodes = G.node;
    auto& adj   = G.adj;

    // 1) 为每个节点在结果字典中放一个空 dict
    std::vector<node_t> ids;
    ids.reserve(nodes.size());
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        node_t u = it->first;
        ids.push_back(u);
        py::object u_obj = G.id_to_node.attr("__getitem__")(py::cast(u));
        result_dict[u_obj] = py::dict();
    }
    if (ids.empty()) return result_dict;

    // 2) 收集边列表 (u,v,w)
    //    若是无向图：仅收集 u < v 侧以去重；若是有向图，请移除 "if (u < v)"。
    struct Edge { node_t u, v; weight_t w; };
    std::vector<Edge> edges;
    for (auto uit = adj.begin(); uit != adj.end(); ++uit) {
        node_t u = uit->first;
        const auto& nbrs = uit->second;
        for (auto vit = nbrs.begin(); vit != nbrs.end(); ++vit) {
            node_t v = vit->first;
            const edge_attr_dict_factory& eattr = vit->second;
            auto w_it = eattr.find("weight");
            weight_t w = (w_it != eattr.end()) ? w_it->second : static_cast<weight_t>(1);

            if (u < v) {            // 无向图去重；有向图请删除此行条件
                edges.push_back({u, v, w});
            }
        }
    }

    // 3) Kruskal：边按权升序
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.w < b.w; });

    // 4) 并查集（DSU）
    //    将 node_t 映射到 0..N-1 索引
    std::unordered_map<node_t, int> idx;
    idx.reserve(ids.size());
    for (int i = 0; i < static_cast<int>(ids.size()); ++i) idx[ids[i]] = i;

    std::vector<int> parent(ids.size()), rank_(ids.size(), 0);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    };

    // 5) 选择 MST 边（以 id->id->weight 暂存）
    std::unordered_map<node_t, std::unordered_map<node_t, weight_t>> res_dict;

    for (const auto& e : edges) {
        auto it_u = idx.find(e.u);
        auto it_v = idx.find(e.v);
        if (it_u == idx.end() || it_v == idx.end()) continue; // 安全性
        if (unite(it_u->second, it_v->second)) {
            // 只写一个方向；若你想对称写入，可在最终输出时补回另一侧
            res_dict[e.u][e.v] = e.w;
        }
    }

    // 6) 将 id -> Python 节点对象，并写回 result_dict
    for (auto it = res_dict.begin(); it != res_dict.end(); ++it) {
        node_t u = it->first;
        py::object u_obj = G.id_to_node.attr("__getitem__")(py::cast(u));
        py::dict u_dict  = result_dict.attr("__getitem__")(u_obj).cast<py::dict>();

        const auto& nbrs = it->second;
        for (auto jt = nbrs.begin(); jt != nbrs.end(); ++jt) {
            node_t v = jt->first;
            weight_t w = jt->second;

            py::object v_obj = G.id_to_node.attr("__getitem__")(py::cast(v));
            u_dict[v_obj] = w;

            // 如需无向对称输出，取消下面两行注释：
            // py::dict v_dict = result_dict.attr("__getitem__")(v_obj).cast<py::dict>();
            // v_dict[u_obj] = w;
        }
    }

    return result_dict;
}
py::dict Floyd(Graph& G) {
    py::dict result_dict;

    auto& nodes = G.node;
    auto& adj   = G.adj;

    // 节点 id 列表
    std::vector<node_t> ids;
    ids.reserve(nodes.size());
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        ids.push_back(it->first);
        // 先为每个节点放一个空 dict
        py::object u_obj = G.id_to_node.attr("__getitem__")(py::cast(it->first));
        result_dict[u_obj] = py::dict();
    }
    if (ids.empty()) return result_dict;

    const weight_t INF = std::numeric_limits<weight_t>::infinity();

    // 距离矩阵（用哈希映射表示）
    std::unordered_map<node_t, std::unordered_map<node_t, weight_t>> dist;
    dist.reserve(ids.size());
    for (node_t u : ids) {
        auto& row = dist[u];                 // 创建一行
        row.reserve(ids.size());
        for (node_t v : ids) {
            // 自环距离 0
            if (u == v) {
                row[v] = static_cast<weight_t>(0);
                continue;
            }
            // 有边则取权重，否则 INF
            weight_t w = INF;
            auto it_u = adj.find(u);
            if (it_u != adj.end()) {
                const auto& nbrs = it_u->second;
                auto it_uv = nbrs.find(v);
                if (it_uv != nbrs.end()) {
                    const auto& eattr = it_uv->second;
                    auto wit = eattr.find("weight");
                    w = (wit != eattr.end()) ? wit->second : static_cast<weight_t>(1);
                }
            }
            row[v] = w;
        }
    }

    // Floyd–Warshall：O(n^3)
    for (node_t k : ids) {
        for (node_t i : ids) {
            // 小优化：如果 i->k 已不可达，跳过内层
            weight_t dik = dist[i][k];
            if (dik == INF) continue;

            for (node_t j : ids) {
                weight_t kj = dist[k][j];
                if (kj == INF) continue;

                weight_t through = static_cast<weight_t>(dik + kj);
                weight_t& ij    = dist[i][j];
                if (through < ij) ij = through;
            }
        }
    }

    // 写出为 {py_node: {py_node: distance}}
    for (const auto& row : dist) {
        node_t u = row.first;
        py::object u_obj = G.id_to_node.attr("__getitem__")(py::cast(u));
        py::dict u_dict  = result_dict.attr("__getitem__")(u_obj).cast<py::dict>();

        for (const auto& kv : row.second) {
            node_t v = kv.first;
            weight_t d = kv.second;
            py::object v_obj = G.id_to_node.attr("__getitem__")(py::cast(v));
            u_dict[v_obj] = d;
        }
    }

    return result_dict;
}
