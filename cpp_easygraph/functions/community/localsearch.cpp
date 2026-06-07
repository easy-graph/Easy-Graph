#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <queue>
#include <map>
#include <random>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include "localsearch.h"

namespace py = pybind11;
using namespace std;

struct RootDecision {
    int superior;
    int path_length;
    int degree;
};

int choose_center(const vector<pair<int, double>>& sorted_multi) {
    if (sorted_multi.size() < 2) {
        return 1;
    }
    
    vector<double> y;
    for (const auto& p : sorted_multi) {
        y.push_back(p.second);
    }
    
    vector<double> delta;
    for (size_t i = 1; i < y.size(); ++i) {
        delta.push_back(fabs(y[i] - y[i-1]));
    }
    
    if (delta.empty()) {
        return 1;
    }
    
    vector<double> delta_nozero;
    for (double d : delta) {
        if (d != 0) {
            delta_nozero.push_back(d);
        }
    }
    
    if (delta_nozero.empty()) {
        return 1;
    }
    
    double mean = 0;
    for (double d : delta_nozero) {
        mean += d;
    }
    mean /= delta_nozero.size();
    
    double variance = 0;
    for (double d : delta_nozero) {
        variance += (d - mean) * (d - mean);
    }
    variance /= delta_nozero.size();
    double std_dev = sqrt(variance);
    
    double threshold = std_dev + mean;
    
    for (size_t i = 0; i < delta.size(); ++i) {
        if (delta[i] > threshold) {
            return i + 1;
        }
    }
    
    return 0;
}

py::object cpp_localsearch(
    py::object G,
    py::object center_num,
    py::object auto_choose_centers,
    py::object maximum_tree,
    py::object seed,
    py::object self_loop
) {
    Graph_generate_linkgraph(G, py::str("weight"));
    Graph& G_ = G.cast<Graph&>();
    Graph_L GL = G_._get_linkgraph_structure();
    py::dict id_to_node = G_.id_to_node;
    
    int n = GL.n;
    if (n == 0) {
        return py::make_tuple(py::none(), py::list(), py::list(), py::dict(), py::dict(), py::none());
    }
    
    bool has_edges = false;
    for (int u = 1; u <= n; ++u) {
        if (GL.head[u] != -1) {
            has_edges = true;
            break;
        }
    }
    if (!has_edges) {
        py::dict result_grouped;
        py::list result_center_dcd;
        py::list result_y_dcd;
        py::dict result_y_partition;
        for (int u = 1; u <= n; ++u) {
            py::object node = id_to_node[py::cast(u)];
            py::list members;
            members.append(node);
            result_grouped[node] = members;
            result_center_dcd.append(node);
            result_y_dcd.append(node);
            result_y_partition[node] = node;
        }
        return py::make_tuple(py::none(), result_center_dcd, result_y_dcd, result_y_partition, result_grouped, py::none());
    }
    
    std::mt19937 rng(42);  
    if (!seed.is_none()) {
        try {
            int seed_value = seed.cast<int>();
            rng.seed(seed_value);
        } catch (...) {
        }
    }
    std::uniform_real_distribution<double> random_dist(0.0, 1.0);
    
    unordered_set<int> selfloop_nodes;
    bool has_selfloop = self_loop.cast<bool>();
    if (!has_selfloop) {
        selfloop_nodes.clear();
    }
    
    vector<int> degree(n + 1, 0);
    for (int u = 1; u <= n; ++u) {
        int deg = 0;
        for (int e = GL.head[u]; e != -1; e = GL.edges[e].next) {
            deg++;
        }
        if (selfloop_nodes.count(u)) {
            degree[u] = deg + 1;
        } else {
            degree[u] = deg;
        }
    }
    
    unordered_map<int, vector<int>> dag_adj;
    unordered_map<int, vector<int>> dag_pred;
    
    for (int v = 1; v <= n; ++v) {
        if (degree[v] == 0) continue;
        int kv = degree[v];
        vector<pair<int, int>> neighbors;
        for (int e = GL.head[v]; e != -1; e = GL.edges[e].next) {
            int nn = GL.edges[e].to;
            if (nn == v && selfloop_nodes.count(v)) continue;
            int deg_nn = degree[nn];
            neighbors.push_back({nn, deg_nn});
        }
        
        if (!neighbors.empty()) {
            int knnmax = -1;
            for (auto& p : neighbors) {
                if (p.second > knnmax) knnmax = p.second;
            }
            
            if (knnmax >= kv) {
                for (auto& p : neighbors) {
                    if (p.second == knnmax) {
                        int nn = p.first;
                        
                        bool already_has = false;
                        bool has_reverse = false;
                        for (int existing : dag_adj[v]) {
                            if (existing == nn) {
                                already_has = true;
                                break;
                            }
                        }
                        
                        for (int existing : dag_adj[nn]) {
                            if (existing == v) {
                                has_reverse = true;
                                break;
                            }
                        }
                        if (!already_has && !has_reverse) {
                            dag_adj[v].push_back(nn);
                            dag_pred[nn].push_back(v);
                        }
                    }
                }
            }
        }
    }
    
    vector<int> out_degree_dag(n + 1, 0);
    for (int u = 1; u <= n; ++u) {
        out_degree_dag[u] = (int)dag_adj[u].size();
    }
    
    vector<int> roots;
    for (int u = 1; u <= n; ++u) {
        if (out_degree_dag[u] == 0 && degree[u] > 0) {
            roots.push_back(u);
        }
    }
    
    if (roots.empty()) {
        for (int u = 1; u <= n; ++u) {
            if (degree[u] > 0) {
                roots.push_back(u);
            }
        }
    }
    
    if (roots.size() > 1) {
        bool all_same_degree = true;
        int first_degree = -1;
        for (int root : roots) {
            if (first_degree == -1) {
                first_degree = degree[root];
            } else if (degree[root] != first_degree) {
                all_same_degree = false;
                break;
            }
        }
        
        if (all_same_degree) {
            int max_root = -1;
            for (int root : roots) {
                if (root > max_root) {
                    max_root = root;
                }
            }
            roots.clear();
            roots.push_back(max_root);
        }
    }
    
    unordered_map<int, int> tree_rootnode;
    unordered_map<int, int> tree_parentnode;
    unordered_map<int, int> tree_distancetoroot;
    
    for (int i = 1; i <= n; ++i) {
        tree_rootnode[i] = -1;
        tree_parentnode[i] = -1;
        tree_distancetoroot[i] = -1;
    }
    
    queue<pair<int, int>> bfs_queue;
    for (int root : roots) {
        bfs_queue.push({root, 0});
        tree_rootnode[root] = root;
        tree_parentnode[root] = -1;
        tree_distancetoroot[root] = 0;
    }
    
    vector<int> visited(n + 1, 0);
    for (int root : roots) {
        visited[root] = 1;
    }
    
    while (!bfs_queue.empty()) {
        int parent, dist;
        std::tie(parent, dist) = bfs_queue.front();
        bfs_queue.pop();
        
        for (int pred : dag_pred[parent]) {
           
            if (tree_distancetoroot[pred] != -1 && tree_distancetoroot[pred] < dist + 1) {
                continue;  
            }
            
            if (tree_distancetoroot[pred] == -1) {
                
                tree_rootnode[pred] = tree_rootnode[parent];
                tree_parentnode[pred] = parent;
                tree_distancetoroot[pred] = dist + 1;
                bfs_queue.push({pred, dist + 1});
            } else if (tree_distancetoroot[pred] == dist + 1) {
                
                if (random_dist(rng) < 0.5) {
                    continue;  
                }
                
                tree_rootnode[pred] = tree_rootnode[parent];
                tree_parentnode[pred] = parent;
            }
        }
    }
    
    unordered_map<int, vector<int>> root_to_node;
    for (int node = 1; node <= n; ++node) {
        int root = tree_rootnode[node];
        if (root != -1) {
            root_to_node[root].push_back(node);
        }
    }
    
    vector<int> valid_roots;
    for (auto& kv : root_to_node) {
        if ((int)kv.second.size() > 1) {
            valid_roots.push_back(kv.first);
        }
    }
    
    unordered_set<int> root_set(valid_roots.begin(), valid_roots.end());
    unordered_map<int, RootDecision> root_decision;
    
    auto BFS_from_s = [&](int s) -> pair<int, int> {
        queue<int> search_queue;
        unordered_map<int, int> path_dict;
        unordered_set<int> seen;
        
        search_queue.push(s);
        seen.insert(s);
        path_dict[s] = 0;
        
        while (!search_queue.empty()) {
            int vertex = search_queue.front();
            search_queue.pop();
            int current_dist = path_dict[vertex];
            
            vector<pair<int, int>> neighbors;
            for (int e = GL.head[vertex]; e != -1; e = GL.edges[e].next) {
                int nn = GL.edges[e].to;
                int deg_nn = degree[nn];
                neighbors.push_back({nn, deg_nn});
            }
            
            sort(neighbors.begin(), neighbors.end(),
                 [](const pair<int, int>& a, const pair<int, int>& b) {
                     return a.second > b.second;
                 });
            
            for (auto& p : neighbors) {
                int w = p.first;
                if (!seen.count(w)) {
                    path_dict[w] = current_dist + 1;
                    seen.insert(w);
                    search_queue.push(w);
                }
                
                if (root_set.count(w) && degree[w] > degree[s]) {
                    return {w, path_dict[w]};
                }
            }
        }
        return {s, -1};
    };
    
    for (int root : valid_roots) {
        auto result = BFS_from_s(root);
        root_decision[root] = {result.first, result.second, degree[root]};
    }
    
    int max_path = -1;
    for (auto& kv : root_decision) {
        if (kv.second.path_length > max_path) {
            max_path = kv.second.path_length;
        }
    }
    if (max_path < 0) max_path = 2;
    
    for (auto& kv : root_decision) {
        if (kv.second.path_length == -1) {
            kv.second.path_length = max_path;
        }
    }
    
    unordered_map<int, RootDecision> node_plot = root_decision;
    for (int node = 1; node <= n; ++node) {
        if (node_plot.find(node) == node_plot.end() && degree[node] > 0) {
            int parent = tree_parentnode[node];
            node_plot[node] = {parent, 1, degree[node]};
        }
    }
    
    vector<int> node_ids;
    vector<int> degrees;
    vector<int> path_lens;
    for (auto& kv : node_plot) {
        node_ids.push_back(kv.first);
        degrees.push_back(kv.second.degree);
        path_lens.push_back(kv.second.path_length);
    }
    
    for (size_t i = 0; i < path_lens.size(); ++i) {
        if (degrees[i] <= 1) {
            path_lens[i] = 1;
        }
    }
    
    unordered_map<int, int> degree_rank;
    vector<pair<int, int>> sorted_by_deg;
    for (size_t i = 0; i < node_ids.size(); ++i) {
        sorted_by_deg.push_back({degrees[i], node_ids[i]});
    }
    sort(sorted_by_deg.begin(), sorted_by_deg.end(),
         [](const pair<int, int>& a, const pair<int, int>& b) {
             return a.first < b.first;
         });
    
    int rank = 1;
    int last_deg = -1;
    for (auto& p : sorted_by_deg) {
        if (p.first != last_deg) {
            degree_rank[p.second] = rank;
            rank++;
            last_deg = p.first;
        } else {
            degree_rank[p.second] = rank - 1;
        }
    }
    
    int min_rank = 1;
    int max_rank = rank - 1;
    double rank_range = (max_rank - min_rank);
    if (rank_range == 0) rank_range = 1;
    
    vector<double> square_path(node_ids.size());
    double max_sq_path = 0;
    double min_sq_path = 1e9;
    for (size_t i = 0; i < node_ids.size(); ++i) {
        square_path[i] = (double)path_lens[i] * (double)path_lens[i];
        if (square_path[i] > max_sq_path) max_sq_path = square_path[i];
        if (square_path[i] < min_sq_path) min_sq_path = square_path[i];
    }
    double sq_range = max_sq_path - min_sq_path;
    if (sq_range == 0) sq_range = 1;
    
    unordered_map<int, double> multi_dict;
    for (size_t i = 0; i < node_ids.size(); ++i) {
        int node = node_ids[i];
        double norm_deg, norm_sq_path;
        
        if (max_rank == min_rank) {
            norm_deg = 1.0 / (double)node_ids.size();
        } else {
            norm_deg = (double)(degree_rank[node] - min_rank) / rank_range;
        }
        
        if (max_sq_path == min_sq_path) {
            norm_sq_path = 1.0 / (double)node_ids.size();
        } else {
            norm_sq_path = (square_path[i] - min_sq_path) / sq_range;
        }
        
        multi_dict[node] = norm_deg * norm_sq_path;
    }
    
    vector<pair<int, double>> sorted_multi;
    for (auto& kv : multi_dict) {
        sorted_multi.push_back(kv);
    }
    sort(sorted_multi.begin(), sorted_multi.end(),
         [](const pair<int, double>& a, const pair<int, double>& b) {
             if (fabs(a.second - b.second) > 1e-9) return a.second > b.second;
             return a.first > b.first;
         });
    
    int num_centers = (int)valid_roots.size();
    
    bool auto_choose = auto_choose_centers.cast<bool>();
    if (auto_choose && sorted_multi.size() > 0) {
        int auto_centernum = choose_center(sorted_multi);
        if (!center_num.is_none()) {
            int user_center_num = center_num.cast<int>();
            num_centers = (auto_centernum < user_center_num) ? auto_centernum : user_center_num;
        } else {
            num_centers = auto_centernum;
        }
    } else if (!center_num.is_none()) {
        num_centers = center_num.cast<int>();
    }
    
    if (num_centers <= 0) {
        num_centers = (int)valid_roots.size();
    }
    
    vector<int> center_dcd;
    int local_cnt = 0;
    for (size_t i = 0; i < sorted_multi.size() && local_cnt < num_centers; ++i) {
        if (sorted_multi[i].second > 0) {
            local_cnt++;
            center_dcd.push_back(sorted_multi[i].first);
        }
    }
    
    if (center_dcd.empty() && !sorted_multi.empty()) {
        center_dcd.push_back(sorted_multi[0].first);
    }
    
    bool all_same_degree = true;
    int first_deg = -1;
    for (int i = 1; i <= n && all_same_degree; ++i) {
        if (degree[i] > 0) {
            if (first_deg == -1) {
                first_deg = degree[i];
            } else if (degree[i] != first_deg) {
                all_same_degree = false;
            }
        }
    }
    
    if (all_same_degree && n > 0) {
        center_dcd.clear();
        center_dcd.push_back(n);
    }
    
    unordered_set<int> center_set(center_dcd.begin(), center_dcd.end());
    
    for (int node : valid_roots) {
        int superior = root_decision[node].superior;
        tree_parentnode[node] = superior;
        tree_rootnode[node] = superior;
    }
    
    for (int node = 0; node < n; ++node) {
        if (degree[node] > 0 && center_set.count(node)) {
            tree_rootnode[node] = node;
        }
    }
    
    for (int node = 0; node < n; ++node) {
        if (degree[node] == 0) continue;
        vector<int> recent;
        recent.push_back(node);
        bool flag = false;
        
        while (center_set.find(tree_rootnode[node]) == center_set.end() && !flag) {
            int j = tree_rootnode[node];
            if (j == -1 || find(recent.begin(), recent.end(), j) != recent.end()) {
                tree_rootnode[node] = -1;
                flag = true;
                break;
            }
            recent.push_back(j);
            tree_rootnode[node] = tree_rootnode[j];
        }
    }
    
    unordered_map<int, int> y_partition;
    vector<int> y_dcd;
    
    for (int node = 1; node <= n; ++node) {
        if (degree[node] == 0) continue;
        int root = tree_rootnode[node];
        if (root == -1 && !center_dcd.empty()) {
            root = center_dcd[0];
            tree_rootnode[node] = root;
        }
        y_partition[node] = root;
        if (root == -1) {
            y_dcd.push_back(-1);
        } else {
            y_dcd.push_back(root);
        }
    }
    
    unordered_map<int, vector<int>> grouped;
    for (auto& kv : y_partition) {
        int center = kv.second;
        if (center != -1) {
            grouped[center].push_back(kv.first);
        }
    }
    
    py::dict result_grouped;
    for (auto& kv : grouped) {
        py::list members;
        for (int node_id : kv.second) {
            members.append(id_to_node[py::cast(node_id)]);
        }
        result_grouped[id_to_node[py::cast(kv.first)]] = members;
    }
    
    py::list result_center_dcd;
    for (int center : center_dcd) {
        result_center_dcd.append(id_to_node[py::cast(center)]);
    }
    
    py::list result_y_dcd;
    for (int label : y_dcd) {
        if (label == -1) {
            result_y_dcd.append(py::cast(-1));
        } else {
            result_y_dcd.append(id_to_node[py::cast(label)]);
        }
    }
    
    py::dict result_y_partition;
    for (auto& kv : y_partition) {
        if (kv.second == -1) {
            result_y_partition[id_to_node[py::cast(kv.first)]] = py::cast(-1);
        } else {
            result_y_partition[id_to_node[py::cast(kv.first)]] = id_to_node[py::cast(kv.second)];
        }
    }
    
    return py::make_tuple(
        py::none(),
        result_center_dcd,
        result_y_dcd,
        result_y_partition,
        result_grouped,
        py::none()
    );
}
