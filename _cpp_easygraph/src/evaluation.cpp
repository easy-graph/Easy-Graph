#include "evaluation.h"
#include "graph.h"
#include "utils.h"

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>()(p.first);
    auto h2 = std::hash<T2>()(p.second);
    return h1 ^ h2;
  }
};

std::unordered_map<std::pair<node_t, node_t>, weight_t, pair_hash> sum_nmw_rec,
    max_nmw_rec, local_constraint_rec;

enum norm_t { sum, max };

weight_t mutual_weight(adj_dict_factory &G, node_t u, node_t v,
                       std::string weight) {
  weight_t a_uv = 0, a_vu = 0;
  if (G.count(u) && G[u].count(v)) {
    edge_attr_dict_factory &guv = G[u][v];
    a_uv = guv.count(weight) ? guv[weight] : 1;
  }
  if (G.count(v) && G[v].count(u)) {
    edge_attr_dict_factory &gvu = G[v][u];
    a_uv = gvu.count(weight) ? gvu[weight] : 1;
  }
  return a_uv + a_vu;
}

weight_t normalized_mutual_weight(adj_dict_factory &G, node_t u, node_t v,
                                  std::string weight, norm_t norm = sum) {
  std::pair<node_t, node_t> edge = std::make_pair(u, v);
  auto &nmw_rec = (norm == sum) ? sum_nmw_rec : max_nmw_rec;
  if (nmw_rec.count(edge)) {
    return nmw_rec[edge];
  } else {
    weight_t scale = 0;
    for (auto &w : G[u]) {
      weight_t temp_weight = mutual_weight(G, u, w.first, weight);
      scale =
          (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
    }
    weight_t nmw = scale ? (mutual_weight(G, u, v, weight) / scale) : 0;
    nmw_rec[edge] = nmw;
    return nmw;
  }
}

weight_t local_constraint(adj_dict_factory &G, node_t u, node_t v,
                          std::string weight = "None") {
  std::pair<node_t, node_t> edge = std::make_pair(u, v);
  if (local_constraint_rec.count(edge)) {
    return local_constraint_rec[edge];
  } else {
    weight_t direct = normalized_mutual_weight(G, u, v, weight);
    weight_t indirect = 0;
    for (auto &w : G[u]) {
      indirect += normalized_mutual_weight(G, u, w.first, weight) *
                  normalized_mutual_weight(G, w.first, v, weight);
    }
    weight_t result = pow((direct + indirect), 2);
    local_constraint_rec[edge] = result;
    return result;
  }
}

std::pair<node_t, weight_t>
compute_constraint_of_v(adj_dict_factory &G, node_t v, std::string weight) {
  weight_t constraint_of_v = 0;
  if (G[v].size() == 0) {
    constraint_of_v = Py_NAN;
  } else {
    for (const auto &n : G[v]) {
      constraint_of_v += local_constraint(G, v, n.first, weight);
    }
  }
  return std::make_pair(v, constraint_of_v);
}

py::dict constraint(const py::object &G, py::object nodes,
                           const py::object &weight,
                           const py::object &n_workers /*未使用但保留签名*/) {
  // 1) 预处理
  std::string weight_key = weight_to_string(weight);
  sum_nmw_rec.clear();
  max_nmw_rec.clear();
  local_constraint_rec.clear();

  // 2) nodes 缺省：从 G.nodes 获取；兼容属性或方法
  if (nodes.is_none()) {
    nodes = G.attr("nodes");
  }
  py::list nodes_list;
  if (py::isinstance<py::function>(nodes) || PyCallable_Check(nodes.ptr()))
    nodes_list = py::list(nodes());
  else
    nodes_list = py::list(nodes);

  // 3) 主循环
  py::dict result;
  Graph &G_ = G.cast<Graph &>();

  for (py::ssize_t i = 0; i < py::len(nodes_list); ++i) {
    py::object v = nodes_list[i];

    // v_id = G_.node_to_id[v]
    // 若 node_to_id 是 py::dict
    node_t v_id = G_.node_to_id[v].cast<node_t>();

    // 计算 (best_id, value)
    std::pair<node_t, weight_t> p =
        compute_constraint_of_v(G_.adj, v_id, weight_key);

    // key = G_.id_to_node[p.first]
    py::object key = G_.id_to_node.attr("get")(py::int_(p.first));

    // result[key] = p.second
    result[key] = py::cast(p.second);
  }

  return result;
}

weight_t redundancy(adj_dict_factory &G, node_t u, node_t v,
                    std::string weight = "None") {
  weight_t r = 0;
  for (const auto &neighbor_info : G[u]) {
    node_t w = neighbor_info.first;
    r += normalized_mutual_weight(G, u, w, weight) *
         normalized_mutual_weight(G, v, w, weight, max);
  }
  return 1 - r;
}

py::dict effective_size(const py::object &G, py::object nodes,
                               const py::object &weight,
                               const py::object &n_workers /*未用，保留签名*/) {
  Graph &G_ = G.cast<Graph &>();

  // 与原逻辑一致的全局缓存清理
  sum_nmw_rec.clear();
  max_nmw_rec.clear();

  py::dict eff;

  // nodes 缺省：迭代整个图
  if (nodes.is_none()) {
    nodes = G; // 假定 G 可迭代出节点
  }
  py::list nodes_list(nodes);

  // 为了兼容 G[v] 的写法
  py::object getitem = G.attr("__getitem__");
  py::object py_nan = py::module_::import("math").attr("nan");

  // 无向图 + 无权：使用 ego_subgraph 的闭式公式
  if (!G.attr("is_directed")().cast<bool>() && weight.is_none()) {
    for (py::ssize_t i = 0; i < py::len(nodes_list); ++i) {
      py::object v = nodes_list[i];

      // 度为 0 → NaN
      if (py::len(getitem(v)) == 0) {
        eff[v] = py_nan;
        continue;
      }

      py::object E = G.attr("ego_subgraph")(v);
      auto n = py::len(E);
      if (n > 1) {
        weight_t size = E.attr("size")().cast<weight_t>();
        // val = n - 1 - (2 * size) / (n - 1)
        double nd = static_cast<double>(n);
        double val = nd - 1.0 - (2.0 * static_cast<double>(size)) / (nd - 1.0);
        eff[v] = py::float_(val);
      } else {
        eff[v] = py::int_(0);
      }
    }
  } else {
    // 有向或有权：按邻接表+redundancy 求和
    const std::string weight_key = weight_to_string(weight);

    for (py::ssize_t i = 0; i < py::len(nodes_list); ++i) {
      py::object v = nodes_list[i];

      if (py::len(getitem(v)) == 0) {
        eff[v] = py_nan;
        continue;
      }

      weight_t redundancy_sum = weight_t(0);

      // v_id = node_to_id[v]
      // （pybind11 的 py::dict 不建议用 operator[] 取值，改用 get）
      node_t v_id = G_.node_to_id.attr("get")(v, py::none()).cast<node_t>();

      for (const auto &nbr : G_.adj[v_id]) {
        node_t u_id = nbr.first;
        redundancy_sum += redundancy(G_.adj, v_id, u_id, weight_key);
      }
      eff[v] = py::cast(redundancy_sum);
    }
  }

  return eff;
}

py::dict hierarchy(Graph& G,
                   py::object nodes,
                   py::object weight,
                   py::object n_workers) {
  sum_nmw_rec.clear();
  max_nmw_rec.clear();
  local_constraint_rec.clear();

  // 权重键（允许 None；你的 local_constraint/取权重逻辑应能在缺失时回退到 1）
  const std::string weight_key = weight_to_string(weight);

  // 构造要处理的节点列表（Python 对象形式）
  std::vector<py::object> nodes_vec;
  if (nodes.is_none()) {
    nodes_vec.reserve(G.node.size());
    for (const auto &kv : G.node) {
      const node_t nid = kv.first;
      // 从 id_to_node 取回 Python 节点对象（const 情况下用 __getitem__）
      nodes_vec.emplace_back(G.id_to_node.attr("__getitem__")(py::cast(nid)));
    }
  } else {
    for (py::handle h : nodes) {
      py::object v = py::reinterpret_borrow<py::object>(h);
      if (G.node_to_id.contains(v))
        nodes_vec.emplace_back(std::move(v));
    }
  }

  py::dict result;

  for (const py::object &v_obj : nodes_vec) {
    // v -> id；若节点不在图中（并发修改等），跳过
    if (!G.node_to_id.contains(v_obj)) {
      result[v_obj] = 0.0;
      continue;
    }
    const node_t v_id = py::cast<node_t>(G.node_to_id[v_obj]);

    // 邻居集合（去掉自环）
    std::vector<node_t> nbrs_vec;
    if (auto it = G.adj.find(v_id); it != G.adj.end()) {
      nbrs_vec.reserve(it->second.size());
      for (const auto &kv : it->second) {
        const node_t w_id = kv.first;
        if (w_id != v_id)
          nbrs_vec.push_back(w_id);
      }
    }
    const int n = static_cast<int>(nbrs_vec.size());
    if (n <= 1) {
      result[v_obj] = 0.0;
      continue;
    }

    // 计算每个邻居的 local_constraint 以及总和 C
    std::vector<double> c_vals;
    c_vals.reserve(n);

    double C = 0.0;
    for (const node_t w_id : nbrs_vec) {
      const double cw =
          static_cast<double>(local_constraint(G.adj, v_id, w_id, weight_key));
      c_vals.push_back(cw);
      C += cw;
    }

    double h_sum = 0.0;
    if (C > 0.0) {
      // hierarchy_sum = sum_{w in N(v)} ( (c_w/C)*n * log((c_w/C)*n) ) / (n*log
      // n)
      const double n_log_n =
          static_cast<double>(n) * std::log(static_cast<double>(n));
      for (double cw : c_vals) {
        const double p = cw / C; // 归一化权重
        const double t = p * static_cast<double>(n);
        if (t > 0.0) {
          h_sum += (t * std::log(t)) / n_log_n;
        }
      }
    }
    result[v_obj] = h_sum; // 若 C==0 则为 0
  }

  return result;
}
