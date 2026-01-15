#include "evaluation.h"
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef EASYGRAPH_ENABLE_GPU
#include <gpu_easygraph.h>
#endif

#include "../../classes/graph.h"
#include "../../classes/directed_graph.h"
#include "../../common/utils.h"

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>()(p.first);
        auto h2 = std::hash<T2>()(p.second);
        return h1 ^ h2;
    }
};

typedef std::unordered_map<std::pair<node_t, node_t>, weight_t, pair_hash> rec_type;

enum norm_t {
    sum,
    max
};

weight_t mutual_weight(const Graph& G, node_t u, node_t v, const std::string& weight) {
    weight_t a_uv = 0, a_vu = 0;
    if (G.adj.count(u)) {
        const auto& adj_u = G.adj.at(u);
        auto it = adj_u.find(v);
        if (it != adj_u.end()) {
            const auto& guv = it->second;
            a_uv = guv.count(weight) ? guv.at(weight) : 1;
        }
    }
    if (G.adj.count(v)) {
        const auto& adj_v = G.adj.at(v);
        auto it = adj_v.find(u);
        if (it != adj_v.end()) {
            const auto& gvu = it->second;
            a_vu = gvu.count(weight) ? gvu.at(weight) : 1;
        }
    }
    return a_uv + a_vu;
}

weight_t directed_mutual_weight(const DiGraph& G, node_t u, node_t v, const std::string& weight) {
    weight_t a_uv = 0, a_vu = 0;
    if (G.adj.count(u)) {
        const auto& adj_u = G.adj.at(u);
        auto it = adj_u.find(v);
        if (it != adj_u.end()) {
            const auto& guv = it->second;
            a_uv = guv.count(weight) ? guv.at(weight) : 1;
        }
    }
    if (G.adj.count(v)) {
        const auto& adj_v = G.adj.at(v);
        auto it = adj_v.find(u);
        if (it != adj_v.end()) {
            const auto& gvu = it->second;
            a_vu = gvu.count(weight) ? gvu.at(weight) : 1;
        }
    }
    return a_uv + a_vu;
}

weight_t normalized_mutual_weight(const Graph& G, node_t u, node_t v, const std::string& weight, norm_t norm) {
    weight_t scale = 0;
    if (G.adj.count(u)) {
        for (const auto& w_pair : G.adj.at(u)) {
            weight_t temp_weight = mutual_weight(G, u, w_pair.first, weight);
            scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
        }
    }
    weight_t nmw = scale ? (mutual_weight(G, u, v, weight) / scale) : 0;
    return nmw;
}

weight_t directed_normalized_mutual_weight(const DiGraph& G, node_t u, node_t v, const std::string& weight, norm_t norm) {
    weight_t scale = 0;
    if (G.adj.count(u)) {
        for (const auto& w_pair : G.adj.at(u)) {
            weight_t temp_weight = directed_mutual_weight(G, u, w_pair.first, weight);
            scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
        }
    }
    if (G.pred.count(u)) {
        for (const auto& w_pair : G.pred.at(u)) {
            weight_t temp_weight = directed_mutual_weight(G, u, w_pair.first, weight);
            scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
        }
    }
    weight_t nmw = scale ? (directed_mutual_weight(G, u, v, weight) / scale) : 0;
    return nmw;
}


void preprocess_graph_for_constraint(
    Graph& G, 
    std::string weight_key,
    std::unordered_map<node_t, std::unordered_map<node_t, double>>& weighted_adj,
    std::unordered_map<node_t, double>& strength
) {
    for (auto& u_entry : G.adj) {
        node_t u = u_entry.first;
        for (auto& v_entry : u_entry.second) {
            node_t v = v_entry.first;
            double w = 1.0;
            if (!weight_key.empty() && v_entry.second.count(weight_key)) {
                w = v_entry.second[weight_key];
            }
            weighted_adj[u][v] += w;
            strength[u] += w;
            weighted_adj[v][u] += w;
            strength[v] += w;
        }
    }
}

py::object invoke_cpp_constraint(py::object G, py::object nodes, py::object weight) {
    std::string weight_key = weight_to_string(weight);

    if (nodes.is_none()) {
        nodes = G.attr("nodes");
    }
    py::list nodes_list = py::list(nodes);
    int nodes_list_len = py::len(nodes_list);
    
    Graph& G_ref = G.cast<Graph&>();
    std::vector<node_t> node_ids(nodes_list_len);
    for (int i = 0; i < nodes_list_len; i++) {
        node_ids[i] = G_ref.node_to_id[nodes_list[i]].cast<node_t>();
    }

    std::unordered_map<node_t, std::unordered_map<node_t, double>> weighted_adj;
    std::unordered_map<node_t, double> strength;
    preprocess_graph_for_constraint(G_ref, weight_key, weighted_adj, strength);

    std::vector<double> constraint_results(nodes_list_len, 0.0);

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nodes_list_len; i++) {
            node_t u = node_ids[i];
            
            auto str_it = strength.find(u);
            if (str_it == strength.end() || str_it->second == 0.0) {
                constraint_results[i] = Py_NAN;
                continue;
            }
            double u_strength = str_it->second;

            auto& neighbors_u = weighted_adj[u];
            if (neighbors_u.empty()) {
                constraint_results[i] = Py_NAN;
                continue;
            }

            std::unordered_map<node_t, double> contrib;

            for (auto& neighbor : neighbors_u) {
                node_t j = neighbor.first;
                double w_uj = neighbor.second;
                double p_uj = w_uj / u_strength;
                
                contrib[j] += p_uj;
            }

            for (auto& neighbor_j : neighbors_u) {
                node_t j = neighbor_j.first;
                double w_uj = neighbor_j.second;
                double p_uj = w_uj / u_strength;

                auto q_it = weighted_adj.find(j);
                if (q_it != weighted_adj.end()) {
                    double j_strength = strength[j];
                    for (auto& neighbor_q : q_it->second) {
                        node_t q = neighbor_q.first;
                        if (q == u) continue;

                        double w_jq = neighbor_q.second;
                        double p_jq = w_jq / j_strength;

                        contrib[q] += p_uj * p_jq;
                    }
                }
            }

            double c_u = 0.0;
            for (auto& neighbor : neighbors_u) {
                node_t j = neighbor.first;
                if (contrib.count(j)) {
                    c_u += pow(contrib[j], 2);
                }
            }
            constraint_results[i] = c_u;
        }
    }

    py::array::ShapeContainer ret_shape{nodes_list_len};
    py::array_t<double> ret(ret_shape, constraint_results.data());
    return ret;
}

#ifdef EASYGRAPH_ENABLE_GPU
static py::object invoke_gpu_constraint(py::object G, py::object nodes, py::object weight) {
    Graph& G_ = G.cast<Graph&>();
    if (weight.is_none()) {
        G_.gen_CSR();
    } else {
        G_.gen_CSR(weight_to_string(weight));
    }
    auto csr_graph = G_.csr_graph;
    auto coo_graph = G_.transfer_csr_to_coo(csr_graph);
    std::vector<int>& V = csr_graph->V;
    std::vector<int>& E = csr_graph->E;
    std::vector<int>& row = coo_graph->row;
    std::vector<int>& col = coo_graph->col;
    std::vector<double> *W_p = weight.is_none() ? &(coo_graph->unweighted_W)
                            : coo_graph->W_map.find(weight_to_string(weight))->second.get();
    std::unordered_map<node_t, int>& node2idx = coo_graph->node2idx;
    int num_nodes = coo_graph->node2idx.size();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    std::vector<double> constraint_results(num_nodes, 0.0);

    std::vector<int> node_mask(num_nodes, 0);
    py::list nodes_list;
    if (!nodes.is_none()) {
        nodes_list = py::list(nodes);
        for (auto node : nodes_list) {
            int node_id = node2idx[G_.node_to_id[node].cast<node_t>()];
            node_mask[node_id] = 1;
        }
    } else {
        nodes_list = py::list(G.attr("nodes"));
        std::fill(node_mask.begin(), node_mask.end(), 1);
    }

    int gpu_r = gpu_easygraph::constraint(V, E, row, col, num_nodes, *W_p, is_directed, node_mask, constraint_results);
    if (gpu_r != gpu_easygraph::EG_GPU_SUCC) {
        py::pybind11_fail(gpu_easygraph::err_code_detail(gpu_r));
    }

    py::array::ShapeContainer ret_shape{(int)constraint_results.size()};
    py::array_t<double> ret(ret_shape, constraint_results.data());

    return ret;
}
#endif

py::object constraint(py::object G, py::object nodes, py::object weight, py::object n_workers) {
#ifdef EASYGRAPH_ENABLE_GPU
    return invoke_gpu_constraint(G, nodes, weight);
#else
    return invoke_cpp_constraint(G, nodes, weight);
#endif
}

weight_t redundancy(const Graph& G, node_t u, node_t v, const std::string& weight) {
    weight_t r = 0;
    if (G.adj.count(v)) {
        for (const auto& n_pair : G.adj.at(v)) {
            node_t w = n_pair.first;
            r += normalized_mutual_weight(G, u, w, weight, sum) * normalized_mutual_weight(G, v, w, weight, max);
        }
    }
    return 1 - r;
}

weight_t directed_redundancy(const DiGraph& G, node_t u, node_t v, const std::string& weight) {
    weight_t r = 0;
    std::unordered_set<node_t> neighbors;
    if (G.adj.count(v)) {
        for (const auto& n : G.adj.at(v)) neighbors.insert(n.first);
    }
    if (G.pred.count(v)) {
        for (const auto& n : G.pred.at(v)) neighbors.insert(n.first);
    }
    
    for (const auto& w : neighbors) {
        r += directed_normalized_mutual_weight(G, u, w, weight, sum) * directed_normalized_mutual_weight(G, v, w, weight, max);
    }
    return 1 - r;
}

py::object invoke_cpp_effective_size(py::object G, py::object nodes, py::object weight) {
    std::string weight_key = weight_to_string(weight); 
    bool is_directed = G.attr("is_directed")().cast<bool>();

    if (nodes.is_none()) {
        nodes = G.attr("nodes");
    }
    py::list nodes_list = py::list(nodes);
    int nodes_list_len = py::len(nodes_list);
    
    Graph& G_ref = G.cast<Graph&>();
    py::object node_to_id = G_ref.node_to_id;
    
    std::vector<node_t> node_ids(nodes_list_len);
    for (int i = 0; i < nodes_list_len; i++) {
        node_ids[i] = node_to_id[nodes_list[i]].cast<node_t>();
    }

    std::vector<double> effective_size_results(nodes_list_len, 0.0);

    bool use_fast_path = !is_directed && (weight.is_none());

    {
        py::gil_scoped_release release;

        if (!is_directed) {
            const Graph& G_ = G.cast<Graph&>();

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < nodes_list_len; i++) {
                node_t v_id = node_ids[i];

                if (G_.adj.find(v_id) == G_.adj.end() || G_.adj.at(v_id).empty()) {
                    effective_size_results[i] = NAN;
                    continue;
                }

                if (use_fast_path) {
                    const auto& v_neighbors = G_.adj.at(v_id);
                    double n = (double)v_neighbors.size();
                    double sum_common = 0;

                    for (const auto& u_pair : v_neighbors) {
                        node_t u = u_pair.first;
                        if (u == v_id) continue; 

                        if (G_.adj.count(u)) {
                            const auto& u_neighbors = G_.adj.at(u);
                            if (v_neighbors.size() < u_neighbors.size()) {
                                for (const auto& w_pair : v_neighbors) {
                                    node_t w = w_pair.first;
                                    if (w == u) continue; 
                                    if (u_neighbors.count(w)) sum_common += 1.0;
                                }
                            } else {
                                for (const auto& w_pair : u_neighbors) {
                                    node_t w = w_pair.first;
                                    if (w == v_id) continue;
                                    if (v_neighbors.count(w)) sum_common += 1.0;
                                }
                            }
                        }
                    }
                    effective_size_results[i] = n - (sum_common / n);

                } else {
                    double redundancy_sum = 0;
                    for (const auto& neighbor_info : G_.adj.at(v_id)) {
                        node_t u_id = neighbor_info.first;
                        redundancy_sum += redundancy(G_, v_id, u_id, weight_key);
                    }
                    effective_size_results[i] = redundancy_sum;
                }
            }
        } else {
            const DiGraph& G_ = G.cast<DiGraph&>();
            
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < nodes_list_len; i++) {
                node_t v_id = node_ids[i];
                
                bool has_neighbors = (G_.adj.count(v_id) && !G_.adj.at(v_id).empty()) || 
                                     (G_.pred.count(v_id) && !G_.pred.at(v_id).empty());
                
                if (!has_neighbors) {
                    effective_size_results[i] = NAN;
                    continue;
                }

                double redundancy_sum = 0;
                if (G_.adj.count(v_id)) {
                    for (const auto& neighbor_info : G_.adj.at(v_id)) {
                        node_t u_id = neighbor_info.first;
                        redundancy_sum += directed_redundancy(G_, v_id, u_id, weight_key);
                    }
                }
                if (G_.pred.count(v_id)) {
                    for (const auto& neighbor_info : G_.pred.at(v_id)) {
                        node_t u_id = neighbor_info.first;
                        redundancy_sum += directed_redundancy(G_, v_id, u_id, weight_key);
                    }
                }
                effective_size_results[i] = redundancy_sum;
            }
        }
    } 

    py::array::ShapeContainer ret_shape{nodes_list_len};
    py::array_t<double> ret(ret_shape, effective_size_results.data());
    return ret;
}

#ifdef EASYGRAPH_ENABLE_GPU
static py::object invoke_gpu_effective_size(py::object G, py::object nodes, py::object weight) {
    Graph& G_ = G.cast<Graph&>();

    if (weight.is_none()) {
        G_.gen_CSR();
    } else {
        G_.gen_CSR(weight_to_string(weight));
    }
    auto csr_graph = G_.csr_graph;
    auto coo_graph = G_.transfer_csr_to_coo(csr_graph);

    std::vector<int>& V = csr_graph->V;
    std::vector<int>& E = csr_graph->E;
    std::vector<int>& row = coo_graph->row;
    std::vector<int>& col = coo_graph->col;

    std::vector<double>* W_p = weight.is_none() ? &(coo_graph->unweighted_W)
                                                : coo_graph->W_map.find(weight_to_string(weight))->second.get();

    std::unordered_map<node_t, int>& node2idx = coo_graph->node2idx;
    int num_nodes = coo_graph->node2idx.size();
    std::vector<double> effective_size_results(num_nodes);
    bool is_directed = G.attr("is_directed")().cast<bool>();

    std::vector<int> node_mask(num_nodes, 0);
    py::list nodes_list;
    if (!nodes.is_none()) {
        nodes_list = py::list(nodes);
        for (auto node : nodes_list) {
            int node_id = node2idx[G_.node_to_id[node].cast<node_t>()];
            node_mask[node_id] = 1;
        }
    } else {
        nodes_list = py::list(G.attr("nodes"));
        std::fill(node_mask.begin(), node_mask.end(), 1);
    }

    int gpu_r = gpu_easygraph::effective_size(V, E, row, col, num_nodes, *W_p, is_directed, node_mask, effective_size_results);

    if (gpu_r != gpu_easygraph::EG_GPU_SUCC) {
        py::pybind11_fail(gpu_easygraph::err_code_detail(gpu_r));
    }

    py::array::ShapeContainer ret_shape{(int)effective_size_results.size()};
    py::array_t<double> ret(ret_shape, effective_size_results.data());

    return ret;
}
#endif

py::object effective_size(py::object G, py::object nodes, py::object weight, py::object n_workers) {
#ifdef EASYGRAPH_ENABLE_GPU
    return invoke_gpu_effective_size(G, nodes, weight);
#else
    return invoke_cpp_effective_size(G, nodes, weight);
#endif
}



py::object efficiency(py::object G, py::object nodes, py::object weight, py::object ignored_arg) {
    return py::none();
}

py::object hierarchy(py::object G, py::object nodes, py::object weight, py::object ignored_arg) {
    return py::none();
}