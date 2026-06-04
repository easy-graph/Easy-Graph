#include "indexed_heap.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include "greedy_modularity.h"

namespace py = pybind11;
using namespace std;

py::object cpp_greedy_modularity_communities(py::object G, py::object weight) {
    Graph& G_ = G.cast<Graph&>();
    
    bool is_directed = G.attr("is_directed")().cast<bool>();
    if (is_directed) {
        throw py::value_error("greedy_modularity_communities currently only supports undirected graphs (Graph class). "
                             "For directed graphs, please use other community detection algorithms.");
    }
    
    string weight_key = weight.is_none() ? "weight" : weight.cast<string>();
    
    Graph_L GL;
    bool used_cached_linkgraph = false;
    
    if (G_.linkgraph_dirty || G_.linkgraph_structure.max_deg == -1) {
        GL = graph_to_linkgraph(G_, false, weight_key, true, false);
        G_.linkgraph_dirty = false;
    } else {
        GL = G_.linkgraph_structure;
        used_cached_linkgraph = true;
    }
    
    int N = GL.n;
    
    if (N == 0) {
        return py::list();
    }
    
    double m = 0.0;
    vector<double> k(N + 1, 0.0);
    
    for (int u = 1; u <= N; ++u) {
        for (int e = GL.head[u]; e != -1; e = GL.edges[e].next) {
            int v = GL.edges[e].to;
            double w = GL.edges[e].w;
            if (v > u) {
                m += w;
            }
            k[u] += w;
        }
    }
    
    if (m == 0) {
        py::list result;
        py::dict id_to_node = G_.id_to_node;
        for (int i = 1; i <= N; ++i) {
            py::set comm;
            comm.add(id_to_node[py::cast(i)]);
            result.append(comm);
        }
        return result;
    }
    
    double q0 = 1.0 / (2.0 * m);
    
    vector<double> a(N);
    for (int i = 0; i < N; ++i) {
        a[i] = k[i + 1] * q0;
    }
    
    vector<int> parent(N);
    vector<vector<int>> nodes(N);
    for (int i = 0; i < N; ++i) {
        parent[i] = i;
        nodes[i].push_back(i + 1);
    }
    
    vector<vector<int>> neigh(N);
    int max_neigh_size = 0;
    for (int i = 0; i < N; ++i) {
        int u = i + 1;
        for (int e = GL.head[u]; e != -1; e = GL.edges[e].next) {
            int v = GL.edges[e].to;
            if (v > 0 && v <= N && v != u) {
                neigh[i].push_back(v - 1);
            }
        }
        sort(neigh[i].begin(), neigh[i].end());
        neigh[i].erase(unique(neigh[i].begin(), neigh[i].end()), neigh[i].end());
        if ((int)neigh[i].size() > max_neigh_size) {
            max_neigh_size = neigh[i].size();
        }
    }
    
    unordered_map<long long, double> edge_weights;
    for (int u = 1; u <= N; ++u) {
        for (int e = GL.head[u]; e != -1; e = GL.edges[e].next) {
            int v = GL.edges[e].to;
            double w = GL.edges[e].w;
            if (v > u) {
                long long key = ((long long)(u - 1) << 32) | (unsigned int)(v - 1);
                edge_weights[key] = w;
            }
        }
    }
    
    unordered_map<long long, double> dq;
    dq.reserve(N * 4);
    
    for (int i = 0; i < N; ++i) {
        int u = i + 1;
        for (int j : neigh[i]) {
            if (j > i) {
                int v = j + 1;
                long long edge_key = ((long long)i << 32) | (unsigned int)j;
                double w = edge_weights.count(edge_key) ? edge_weights[edge_key] : 0.0;
                double dq_val = 2.0 * w * q0 - 2.0 * k[u] * k[v] * q0 * q0;
                long long key = ((long long)i << 32) | (unsigned int)j;
                dq[key] = dq_val;
            }
        }
    }
    
    IndexedMaxHeap H(N);
    for (const auto& kv : dq) {
        int i = (int)(kv.first >> 32);
        int j = (int)(kv.first & 0xFFFFFFFF);
        H.push(kv.second, i, j);
    }
    
    vector<char> merged(N, 0);
    int merge_count = 0;
    vector<int> combined;
    combined.reserve(max_neigh_size * 2);
    
    while (!H.empty()) {
        auto best = H.pop();
        
        int i = best.i;
        int j = best.j;
        
        if (merged[i] || merged[j]) {
            continue;
        }
        if (parent[i] != i || parent[j] != j) {
            continue;
        }
        
        long long key = ((long long)i << 32) | (unsigned int)j;
        auto it = dq.find(key);
        if (it == dq.end() || it->second != best.dq) {
            continue;
        }
        if (best.dq <= 0) {
            break;
        }
        
        nodes[i].insert(nodes[i].end(), nodes[j].begin(), nodes[j].end());
        nodes[j].clear();
        parent[j] = i;
        merged[j] = 1;
        merge_count++;
        
        combined.clear();
        
        const vector<int>& list_i = neigh[i];
        const vector<int>& list_j = neigh[j];
        size_t p1 = 0, p2 = 0;
        
        while (p1 < list_i.size() && p2 < list_j.size()) {
            int val1 = list_i[p1];
            int val2 = list_j[p2];
            if (val1 < val2) {
                if (val1 != i && val1 != j && !merged[val1]) {
                    combined.push_back(val1);
                }
                p1++;
            } else if (val1 > val2) {
                if (val2 != i && val2 != j && !merged[val2]) {
                    combined.push_back(val2);
                }
                p2++;
            } else {
                if (val1 != i && val1 != j && !merged[val1]) {
                    combined.push_back(val1);
                }
                p1++;
                p2++;
            }
        }
        while (p1 < list_i.size()) {
            int val = list_i[p1];
            if (val != i && val != j && !merged[val]) {
                combined.push_back(val);
            }
            p1++;
        }
        while (p2 < list_j.size()) {
            int val = list_j[p2];
            if (val != i && val != j && !merged[val]) {
                combined.push_back(val);
            }
            p2++;
        }
        
        neigh[i].swap(combined);
        
        for (int k_node : neigh[i]) {
            if (k_node == i) continue;
            if (parent[k_node] != k_node) continue;
            
            long long key_ik = ((long long)min(i, k_node) << 32) | (unsigned int)max(i, k_node);
            long long key_jk = ((long long)min(j, k_node) << 32) | (unsigned int)max(j, k_node);
            
            bool has_ik = dq.find(key_ik) != dq.end();
            bool has_jk = dq.find(key_jk) != dq.end();
            
            double new_dq;
            if (has_ik && has_jk) {
                new_dq = dq[key_ik] + dq[key_jk];
            } else if (has_jk) {
                new_dq = dq[key_jk] - 2.0 * a[i] * a[k_node];
            } else {
                new_dq = dq[key_ik] - 2.0 * a[j] * a[k_node];
            }
            
            dq[key_ik] = new_dq;
            if (has_jk) {
                dq.erase(key_jk);
            }
            
            if (H.get_index(i, k_node) >= 0) {
                H.update(new_dq, i, k_node);
            } else {
                H.push(new_dq, i, k_node);
            }
        }
        
        const vector<int>& old_j_neigh = neigh[j];
        for (int k_node : old_j_neigh) {
            if (k_node == i || k_node == j) continue;
            if (parent[k_node] != k_node) continue;
            H.remove(j, k_node);
        }
        
        neigh[j].clear();
        
        a[i] += a[j];
        a[j] = 0;
    }
    
    py::dict id_to_node = G_.id_to_node;
    py::list result;
    
    for (int i = 0; i < N; ++i) {
        if (parent[i] == i && !nodes[i].empty()) {
            py::set comm;
            for (int node_id : nodes[i]) {
                comm.add(id_to_node[py::cast(node_id)]);
            }
            result.append(comm);
        }
    }
    
    py::list sorted_result;
    vector<int> sizes;
    for (size_t i = 0; i < result.size(); ++i) {
        sizes.push_back(py::len(result[i]));
    }
    
    vector<int> indices(result.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&](int a_idx, int b_idx) {
        return sizes[a_idx] > sizes[b_idx];
    });
    
    for (int idx : indices) {
        sorted_result.append(result[idx]);
    }
    
    return sorted_result;
}