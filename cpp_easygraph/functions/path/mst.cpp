#include "mst.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>

#include "../../classes/graph.h"
#include "../../common/utils.h"

UnionFind::UnionFind() {}

UnionFind::UnionFind(std::vector<node_t> elements) {
    for (node_t x : elements) {
        parents[x] = x;
        weights[x] = 1;
    }
}
node_t UnionFind::operator[](node_t object) {
    if (!parents.count(object)) {
        parents[object] = object;
        weights[object] = 1;
        return object;
    }

    std::vector<node_t> path;
    path.push_back(object);
    node_t root = parents[object];
    while (root != path.back()) {
        path.push_back(root);
        root = parents[root];
    }
    for (node_t ancestor : path) {
        parents[ancestor] = root;
    }
    return root;
}

void UnionFind::_union(node_t object1, node_t object2) {
    node_t root, r;
    object1 = (*this)[object1];
    object2 = (*this)[object2];
    if (weights[object1] < weights[object2]) {
        root = object1, r = object2;
    } else {
        root = object2, r = object1;
    }
    weights[root] += weights[r];
    parents[r] = root;
}

struct mst_Edge {
    double wt;
    node_t start_node, end_node;
    edge_attr_dict_factory edge_attr;
    mst_Edge(double wt, node_t start_node, node_t end_node, edge_attr_dict_factory edge_attr) {
        this->wt = wt;
        this->start_node = start_node;
        this->end_node = end_node;
        this->edge_attr = edge_attr;
    }
};

py::object kruskal_mst_edges(py::object G, py::object minimum, py::object weight, py::object data, py::object ignore_nan) {
    UnionFind subtrees;
    Graph G_ = G.cast<Graph&>();
    std::string weight_key = weight_to_string(weight);
    std::vector<std::pair<weight_t, graph_edge>> edges;
    int sign = minimum.cast<py::bool_>().equal(py::cast(true)) ? 1 : -1;
    for (graph_edge& edge : G_._get_edges()) {
        weight_t wt = (edge.attr.count(weight_key) ? edge.attr[weight_key] : 1) * sign;
        if (!ignore_nan.cast<py::bool_>() && isnan(wt)) {
            PyErr_Format(PyExc_ValueError, "NaN found as an edge weight. Edge (%R, %R, %R)", G_.id_to_node[py::cast(edge.u)].ptr(), G_.id_to_node[py::cast(edge.v)].ptr(), attr_to_dict(edge.attr).ptr());
            return py::none();
        }
        edges.emplace_back(wt, edge);
    }
    std::sort(edges.begin(), edges.end(), [](const std::pair<weight_t, graph_edge>& edge1, const std::pair<weight_t, graph_edge>& edge2) -> bool {
        return edge1.first < edge2.first;
    });
    py::list ret;
    for (const auto& edge : edges) {
        node_t u = edge.second.u, v = edge.second.v;
        if (subtrees[u] != subtrees[v]) {
            if (data.cast<bool>()) {
                ret.append(py::make_tuple(G_.id_to_node[py::cast(u)], G_.id_to_node[py::cast(v)], attr_to_dict(edge.second.attr)));
            } else {
                ret.append(py::make_tuple(G_.id_to_node[py::cast(u)], G_.id_to_node[py::cast(v)]));
            }
            subtrees._union(u, v);
        }
    }
    return ret;
};

struct cmp {
    bool operator()(const mst_Edge& node1, const mst_Edge& node2) {
        return node1.wt > node2.wt;
    }
};

py::object prim_mst_edges(py::object G, py::object minimum, py::object weight, py::object data, py::object ignore_nan) {
    Graph& G_ = G.cast<Graph&>();
    py::list res = py::list();
    node_dict_factory nodes_list = G_.node;
    std::unordered_set<node_t> nodes;
    for (node_dict_factory::iterator iter = nodes_list.begin(); iter != nodes_list.end(); iter++) {
        node_t node_id = iter->first;
        nodes.emplace(node_id);
    }
    int sign = 1;
    if (!minimum.cast<py::bool_>().equal(py::cast(true))) {
        sign = -1;
    }
    while (!nodes.empty()) {
        const node_t u = *(nodes.begin());
        nodes.erase(nodes.begin());
        std::priority_queue<mst_Edge, std::vector<mst_Edge>, cmp> frontier;
        std::unordered_map<node_t, bool> visited;
        node_t u_ = u;
        visited.emplace(u_, true);
        adj_attr_dict_factory u_neighbors = G_.adj[u];
        for (adj_attr_dict_factory::iterator i = u_neighbors.begin(); i != u_neighbors.end(); i++) {
            node_t v = i->first;
            edge_attr_dict_factory d = i->second;
            double wt = sign;
            if (d.find(py::cast<std::string>(weight)) != d.end()) {
                wt = d[py::cast<std::string>(weight)] * sign;
            }
            if (isnan(wt)) {
                if (ignore_nan.cast<bool>()) {
                    continue;
                }
                PyErr_Format(PyExc_ValueError, "NaN found as an edge weight. Edge {(%R %R %R)}", (G_.id_to_node.attr("get")(u)).ptr(), G_.id_to_node.attr("get")(v).ptr(), attr_to_dict(d).ptr());
                return py::none();
            }
            frontier.push(mst_Edge(wt, u_, v, d));
        }
        while (!frontier.empty()) {
            mst_Edge node = frontier.top();
            frontier.pop();
            double W = node.wt;
            node_t u_id = node.start_node;
            node_t v_id = node.end_node;
            edge_attr_dict_factory d = node.edge_attr;
            if (visited.find(v_id) != visited.end() || nodes.find(v_id) == nodes.end()) {
                continue;
            }
            if (data.cast<bool>()) {
                res.append(py::make_tuple(G_.id_to_node.attr("get")(u_id), G_.id_to_node.attr("get")(v_id), attr_to_dict(d)));
            } else {
                res.append(py::make_tuple(G_.id_to_node.attr("get")(u_id), G_.id_to_node.attr("get")(v_id)));
            }
            visited.emplace(v_id, true);
            nodes.erase(v_id);
            adj_attr_dict_factory v_neighbors = G_.adj[v_id];
            for (adj_attr_dict_factory::iterator j = v_neighbors.begin(); j != v_neighbors.end(); j++) {
                node_t w = j->first;
                edge_attr_dict_factory d2 = j->second;
                if (visited.find(w) != visited.end()) {
                    continue;
                }
                double new_weight = sign;
                if (d2.find(py::cast<std::string>(weight)) != d2.end()) {
                    new_weight = d2[py::cast<std::string>(weight)] * sign;
                }
                frontier.push(mst_Edge(new_weight, v_id, w, d2));
            }
        }
    }
    return res;
}

struct IntUnionFind {
    std::vector<int> parent;
    int component_count;

    IntUnionFind(int n) : component_count(n) {
        parent.resize(n);
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int i) {
        int root = i;
        while (root != parent[root]) root = parent[root];
        
        // 路径压缩：平摊复杂度 O(α(N))
        int curr = i;
        while (curr != root) {
            int nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    }

    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
            component_count--;
        }
    }
};

py::object boruvka_mst_edges(py::object G, py::object minimum, py::object weight, py::object data, py::object ignore_nan) {
    Graph& G_ = G.cast<Graph&>();
    std::string weight_key = weight_to_string(weight);
    int sign = minimum.cast<py::bool_>().equal(py::cast(true)) ? 1 : -1;
    bool return_data = data.cast<bool>();
    bool ignore_n = ignore_nan.cast<bool>();

    // 步骤 A：映射 Python 节点 ID 到连续的 C++ 整数 [0, V-1]
    std::unordered_map<node_t, int> node_to_int;
    for (auto const& item : G_.node) {
        node_t node_id = item.first;
        node_to_int[node_id] = node_to_int.size();
    }
    int num_nodes = node_to_int.size();
    // 步骤 B：边集扁平化 (SoA - Structure of Arrays)
    // original_edges 用于保留 Python 属性；另外三个 vector 专供 OpenMP 狂飙
    std::vector<graph_edge> original_edges; 
    std::vector<int> edge_u;
    std::vector<int> edge_v;
    std::vector<double> edge_wt;

    for (graph_edge& edge : G_._get_edges()) {
        double wt = (edge.attr.count(weight_key) ? edge.attr[weight_key] : 1.0) * sign;
        
        if (std::isnan(wt)) {
            if (!ignore_n) {
                PyErr_Format(PyExc_ValueError, "NaN found as an edge weight. Edge attributes: %R", attr_to_dict(edge.attr).ptr());
                return py::none();
            }
            continue;
        }
        
        original_edges.push_back(edge);
        edge_u.push_back(node_to_int[edge.u]);
        edge_v.push_back(node_to_int[edge.v]);
        edge_wt.push_back(wt);
    }

    int num_edges = original_edges.size();
    IntUnionFind uf(num_nodes);
    std::vector<bool> in_mst(num_edges, false); // 记录被选中边的索引
    
    // 步骤 C：核心并行计算 (释放 GIL，进入纯 C++ 多核域)
    {
        py::gil_scoped_release release;
        
        bool added_edges = true;
        // 只要还有多个连通分量，并且上一轮有新边加入，就继续循环
        while (uf.component_count > 1 && added_edges) {
            added_edges = false;
            
            // cheapest_edge 存的是“边的索引 e”，初始化为 -1
            std::vector<int> cheapest_edge(num_nodes, -1);

            #pragma omp parallel
            {
                // 线程本地的最短边记录，避免多线程写冲突
                std::vector<int> local_cheapest(num_nodes, -1);

                // 无锁遍历所有边
                #pragma omp for nowait
                for (int e = 0; e < num_edges; e++) {
                    int u = edge_u[e];
                    int v = edge_v[e];
                    double wt = edge_wt[e];

                    int set_u = uf.find(u);
                    int set_v = uf.find(v);

                    if (set_u != set_v) {
                        // 比较并更新 u 所在分量的最小边
                        if (local_cheapest[set_u] == -1 || wt < edge_wt[local_cheapest[set_u]]) {
                            local_cheapest[set_u] = e;
                        }
                        // 比较并更新 v 所在分量的最小边
                        if (local_cheapest[set_v] == -1 || wt < edge_wt[local_cheapest[set_v]]) {
                            local_cheapest[set_v] = e;
                        }
                    }
                }

                // 归约：将本地的最短边合并到全局数组
                #pragma omp critical
                {
                    for (int i = 0; i < num_nodes; i++) {
                        if (local_cheapest[i] != -1) {
                            if (cheapest_edge[i] == -1 || edge_wt[local_cheapest[i]] < edge_wt[cheapest_edge[i]]) {
                                cheapest_edge[i] = local_cheapest[i];
                            }
                        }
                    }
                }
            } // #pragma omp parallel 结束

            // 顺序合并本轮找到的最短边
            for (int i = 0; i < num_nodes; i++) {
                if (cheapest_edge[i] != -1) {
                    int e = cheapest_edge[i];
                    int set_u = uf.find(edge_u[e]);
                    int set_v = uf.find(edge_v[e]);

                    if (set_u != set_v) {
                        uf.unite(set_u, set_v);
                        in_mst[e] = true;
                        added_edges = true;
                    }
                }
            }
        }
    } // 重新获取 GIL

    // 步骤 D：利用保留的 original_edges 还原 Python 对象并返回
    py::list ret;
    for (int e = 0; e < num_edges; e++) {
        if (in_mst[e]) {
            const graph_edge& edge = original_edges[e];
            py::object u_obj = G_.id_to_node[py::cast(edge.u)];
            py::object v_obj = G_.id_to_node[py::cast(edge.v)];
            
            if (return_data) {
                ret.append(py::make_tuple(u_obj, v_obj, attr_to_dict(edge.attr)));
            } else {
                ret.append(py::make_tuple(u_obj, v_obj));
            }
        }
    }

    return ret;
}