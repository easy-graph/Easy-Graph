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

struct CompactEdge {
    int u, v, id;
    double wt;
};

struct AtomicBest {
    std::atomic<int> edge_idx;
    AtomicBest() : edge_idx(-1) {}
    AtomicBest(const AtomicBest& other) : edge_idx(other.edge_idx.load()) {}
};

struct IntUnionFind {
    std::vector<int> parent;
    std::vector<int> rank;
    int component_count;

    IntUnionFind(int n) : parent(n), rank(n, 0), component_count(n) {
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]); 
    }

    int find_readonly(int i) const {
        while (i != parent[i]) {
            i = parent[i];
        }
        return i;
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            if (rank[root_i] < rank[root_j]) parent[root_i] = root_j;
            else if (rank[root_i] > rank[root_j]) parent[root_j] = root_i;
            else { parent[root_i] = root_j; rank[root_j]++; }
            component_count--;
            return true;
        }
        return false;
    }
};

py::object boruvka_mst_edges(py::object G, py::object minimum, py::object weight, py::object data, py::object ignore_nan) {
    Graph& G_ = G.cast<Graph&>();
    std::string weight_key = weight_to_string(weight);
    int sign = minimum.cast<bool>() ? 1 : -1;
    bool return_data = data.cast<bool>();
    bool ignore_n = ignore_nan.cast<bool>();

    std::shared_ptr<COOGraph> coo = G_.gen_COO(weight_key);
    int num_nodes = coo->nodes.size();
    int num_edges = coo->row.size();

    if (num_nodes == 0) return py::list();

    std::vector<CompactEdge> active_edges;
    active_edges.reserve(num_edges);
    
    const auto& W_vec = *(coo->W_map[weight_key]); 

    for (int i = 0; i < num_edges; ++i) {
        double wt = W_vec[i] * sign;
        if (std::isnan(wt)) {
            if (!ignore_n) {
                PyErr_Format(PyExc_ValueError, "NaN found as an edge weight.");
                return py::none();
            }
            continue;
        }
        active_edges.push_back({coo->row[i], coo->col[i], i, wt});
    }

    IntUnionFind uf(num_nodes);
    std::vector<bool> in_mst(num_edges, false);
    {
        py::gil_scoped_release release;
        
        while (uf.component_count > 1) {
            std::vector<AtomicBest> best_at(num_nodes);
            bool changed = false;

            #pragma omp parallel for
            for (int i = 0; i < (int)active_edges.size(); ++i) {
                int root_u = uf.find_readonly(active_edges[i].u);
                int root_v = uf.find_readonly(active_edges[i].v);   

                if (root_u != root_v) {
                    auto update_best = [&](int root, int edge_idx) {
                        int current = best_at[root].edge_idx.load(std::memory_order_relaxed);
                        while (current == -1 || active_edges[edge_idx].wt < active_edges[current].wt) {
                            if (best_at[root].edge_idx.compare_exchange_weak(current, edge_idx)) break;
                        }
                    };
                    update_best(root_u, i);
                    update_best(root_v, i);
                }
            }

            for (int i = 0; i < num_nodes; ++i) {
                int e_idx = best_at[i].edge_idx.load();
                if (e_idx != -1) {
                    const auto& e = active_edges[e_idx];
                    if (uf.unite(e.u, e.v)) {
                        in_mst[e.id] = true;
                        changed = true;
                    }
                }
            }

            if (!changed) break;

            auto new_end = std::remove_if(active_edges.begin(), active_edges.end(), [&](const CompactEdge& e) {
                return uf.find(e.u) == uf.find(e.v);
            });
            active_edges.erase(new_end, active_edges.end());
        }
    }

    py::list ret;
    for (int i = 0; i < num_edges; ++i) {
        if (in_mst[i]) {
            node_t u = coo->nodes[coo->row[i]];
            node_t v = coo->nodes[coo->col[i]];
            
            py::object u_obj = G_.id_to_node[py::cast(u)];
            py::object v_obj = G_.id_to_node[py::cast(v)];

            if (return_data) {
                const auto& edge_attr = G_.adj.at(u).at(v);
                ret.append(py::make_tuple(u_obj, v_obj, attr_to_dict(edge_attr)));
            } else {
                ret.append(py::make_tuple(u_obj, v_obj));
            }
        }
    }

    return ret;
}