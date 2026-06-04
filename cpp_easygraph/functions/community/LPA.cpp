#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <queue>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../classes/graph.h"
#include "../../classes/linkgraph.h"
#include "../../common/utils.h"

namespace py = pybind11;
using namespace std;

py::object cpp_LPA(py::object G) {
    Graph& G_ = G.cast<Graph&>();
    Graph_L linkgraph = G_._get_linkgraph_structure();
    int n = linkgraph.n;
    
    py::dict id_to_node = G_.id_to_node;
    
    if (n <= 1) {
        py::dict result;
        if (n == 1) {
            py::list node_list;
            node_list.append(id_to_node[py::cast(1)]);
            result[py::cast(1)] = node_list;
        }
        return result;
    }
    
    vector<vector<int>> adj(n + 1);
    for (int u = 1; u <= n; ++u) {
        for (int e = linkgraph.head[u]; e != -1; e = linkgraph.edges[e].next) {
            int v = linkgraph.edges[e].to;
            if (u != v) {
                adj[u].push_back(v);
            }
        }
    }
    
    for (int u = 1; u <= n; ++u) {
        if (adj[u].size() > 1) {
            sort(adj[u].begin(), adj[u].end());
            adj[u].erase(unique(adj[u].begin(), adj[u].end()), adj[u].end());
        }
    }

    vector<int> labels(n);
    vector<int> nodes(n);
    for (int i = 0; i < n; ++i) {
        labels[i] = i;
        nodes[i] = i + 1;
    }
    
    random_device rd;
    mt19937 gen(rd());
    
    const int MAX_ITERATIONS = 1000;
    int iteration_count = 0;
    
    vector<int> label_count(n, 0);
    vector<int> active_labels;
    active_labels.reserve(128);
    
    while (iteration_count < MAX_ITERATIONS) {
        iteration_count++;
        
        shuffle(nodes.begin(), nodes.end(), gen);
        
        bool changed = false;
        for (int node : nodes) {
            int max_count = 0;
            
            for (int neighbor : adj[node]) {
                int neighbor_label = labels[neighbor - 1];
                if (label_count[neighbor_label] == 0) {
                    active_labels.push_back(neighbor_label);
                }
                label_count[neighbor_label]++;
                if (label_count[neighbor_label] > max_count) {
                    max_count = label_count[neighbor_label];
                }
            }
            
            if (max_count > 0) {
                int current_label = labels[node - 1];
                
                vector<int> best_labels;
                for (int label : active_labels) {
                    if (label_count[label] == max_count) {
                        best_labels.push_back(label);
                    }
                }
                
                bool current_is_best = false;
                for (int label : best_labels) {
                    if (label == current_label) {
                        current_is_best = true;
                        break;
                    }
                }
                
                if (!current_is_best) {
                    changed = true;
                    
                    if (best_labels.size() == 1) {
                        labels[node - 1] = best_labels[0];
                    } else {
                        uniform_int_distribution<> dis(0, best_labels.size() - 1);
                        labels[node - 1] = best_labels[dis(gen)];
                    }
                }
            }
            
            for (int label : active_labels) {
                label_count[label] = 0;
            }
            active_labels.clear();
        }
        
        if (!changed) {
            break;
        }
    }
    
    unordered_map<int, vector<int>> label_to_nodes;
    for (int i = 0; i < n; ++i) {
        int label = labels[i];
        label_to_nodes[label].push_back(i + 1);  
    }
    
    py::dict result;
    int community_id = 1;
    for (const auto& pair : label_to_nodes) {
        py::list node_list;
        for (int internal_id : pair.second) {
            node_list.append(id_to_node[py::cast(internal_id)]);
        }
        result[py::cast(community_id)] = node_list;
        community_id++;
    }
    
    return result;
}