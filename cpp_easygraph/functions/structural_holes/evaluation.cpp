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

template <typename MapType>
inline weight_t get_edge_weight(const MapType& attrs, const std::string& weight_key) {
    if (weight_key.empty()) return 1.0;
    auto it = attrs.find(weight_key);
    return it != attrs.end() ? it->second : 1.0;
}

inline weight_t compute_mutual_weight(const Graph& G, node_t u, node_t v, const std::string& weight_key) {
    weight_t w = 0;
    if (G.adj.count(u)) {
        const auto& adj_u = G.adj.at(u);
        auto it = adj_u.find(v);
        if (it != adj_u.end()) w += get_edge_weight(it->second, weight_key);
    }
    if (G.adj.count(v)) {
        const auto& adj_v = G.adj.at(v);
        auto it = adj_v.find(u);
        if (it != adj_v.end()) w += get_edge_weight(it->second, weight_key);
    }
    return w;
}

inline weight_t compute_directed_mutual_weight(const DiGraph& G, node_t u, node_t v, const std::string& weight_key) {
    weight_t w = 0;
    if (G.adj.count(u)) {
        const auto& adj_u = G.adj.at(u);
        auto it = adj_u.find(v);
        if (it != adj_u.end()) w += get_edge_weight(it->second, weight_key);
    }
    if (G.adj.count(v)) {
        const auto& adj_v = G.adj.at(v);
        auto it = adj_v.find(u);
        if (it != adj_v.end()) w += get_edge_weight(it->second, weight_key);
    }
    return w;
}

std::vector<double> compute_redundancy_core(py::object G_obj, const std::vector<node_t>& target_nodes, const std::string& weight_key, bool is_directed) {
    
    // Cast to C++ objects once to avoid Python API overhead
    const Graph* G_ptr = nullptr;
    const DiGraph* DiG_ptr = nullptr;
    if (is_directed) {
        DiG_ptr = &G_obj.cast<const DiGraph&>();
    } else {
        G_ptr = &G_obj.cast<const Graph&>();
    }

    // Pre-compute max ID and node list
    node_t max_graph_id = 0;
    std::vector<node_t> all_nodes_vec;

    if (is_directed) {
        for (const auto& kv : DiG_ptr->adj) if (kv.first > max_graph_id) max_graph_id = kv.first;
        for (const auto& kv : DiG_ptr->pred) if (kv.first > max_graph_id) max_graph_id = kv.first;
        all_nodes_vec.reserve(DiG_ptr->adj.size() + DiG_ptr->pred.size());
        for(const auto& kv : DiG_ptr->adj) all_nodes_vec.push_back(kv.first);
        for(const auto& kv : DiG_ptr->pred) all_nodes_vec.push_back(kv.first);
    } else {
        for (const auto& kv : G_ptr->adj) if (kv.first > max_graph_id) max_graph_id = kv.first;
        all_nodes_vec.reserve(G_ptr->adj.size());
        for(const auto& kv : G_ptr->adj) all_nodes_vec.push_back(kv.first);
    }

    // Deduplicate nodes
    std::sort(all_nodes_vec.begin(), all_nodes_vec.end());
    all_nodes_vec.erase(std::unique(all_nodes_vec.begin(), all_nodes_vec.end()), all_nodes_vec.end());
    
    // Ensure vector size covers target nodes
    if (!target_nodes.empty()) {
        node_t max_target = *std::max_element(target_nodes.begin(), target_nodes.end());
        max_graph_id = std::max(max_graph_id, max_target);
    }

    // Pre-compute Scale
    std::vector<double> scale_sum_vec(max_graph_id + 1, 0.0);
    std::vector<double> scale_max_vec(max_graph_id + 1, 0.0);

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < (int)all_nodes_vec.size(); ++i) {
        node_t u = all_nodes_vec[i];
        double s_sum = 0;
        double s_max = 0;

        if (is_directed) {
            if (DiG_ptr->adj.count(u)) {
                for(const auto& p : DiG_ptr->adj.at(u)) {
                    weight_t tw = compute_directed_mutual_weight(*DiG_ptr, u, p.first, weight_key);
                    s_sum += tw; s_max = std::max(s_max, (double)tw);
                }
            }
            if (DiG_ptr->pred.count(u)) {
                for(const auto& p : DiG_ptr->pred.at(u)) {
                    weight_t tw = compute_directed_mutual_weight(*DiG_ptr, u, p.first, weight_key);
                    s_sum += tw; s_max = std::max(s_max, (double)tw);
                }
            }
        } else {
            if (G_ptr->adj.count(u)) {
                for(const auto& p : G_ptr->adj.at(u)) {
                    weight_t tw = compute_mutual_weight(*G_ptr, u, p.first, weight_key);
                    s_sum += tw; s_max = std::max(s_max, (double)tw);
                }
            }
        }
        if (u < scale_sum_vec.size()) {
            scale_sum_vec[u] = s_sum;
            scale_max_vec[u] = s_max;
        }
    }

    // Compute Redundancy
    std::vector<double> results(target_nodes.size());

    if (!is_directed) {
        // Undirected
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)target_nodes.size(); i++) {
            node_t v_id = target_nodes[i];
            
            if (G_ptr->adj.find(v_id) == G_ptr->adj.end() || G_ptr->adj.at(v_id).empty()) {
                results[i] = NAN;
                continue;
            }

            const auto& v_neighbors = G_ptr->adj.at(v_id);
            double redundancy_sum = 0;
            double scale_v_sum = (v_id < scale_sum_vec.size()) ? scale_sum_vec[v_id] : 0;

            // Direct iteration avoids malloc locks
            for (const auto& neighbor_info : v_neighbors) {
                node_t u_id = neighbor_info.first;
                double scale_u_max = (u_id < scale_max_vec.size()) ? scale_max_vec[u_id] : 0;
                double r_vu = 0;

                for (const auto& w_pair : v_neighbors) {
                    node_t w_id = w_pair.first;
                    if (u_id == w_id) continue;

                    weight_t mw_uw = compute_mutual_weight(*G_ptr, u_id, w_id, weight_key);
                    if (mw_uw == 0) continue;

                    weight_t mw_vw = compute_mutual_weight(*G_ptr, v_id, w_id, weight_key);

                    double p_iq = (scale_v_sum > 0) ? (mw_vw / scale_v_sum) : 0;
                    double m_jq = (scale_u_max > 0) ? (mw_uw / scale_u_max) : 0;

                    r_vu += p_iq * m_jq;
                }
                redundancy_sum += (1.0 - r_vu);
            }
            results[i] = redundancy_sum;
        }
    } else {
        //Directed
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < target_nodes.size(); i++) {
            node_t v_id = target_nodes[i];

            bool has_neighbors = (DiG_ptr->adj.count(v_id) && !DiG_ptr->adj.at(v_id).empty()) || 
                                 (DiG_ptr->pred.count(v_id) && !DiG_ptr->pred.at(v_id).empty());
            
            if (!has_neighbors) {
                results[i] = NAN;
                continue;
            }

            double redundancy_sum = 0;
            double scale_v_sum = (v_id < scale_sum_vec.size()) ? scale_sum_vec[v_id] : 0;

            // Prepare common candidates
            std::vector<node_t> common_candidates;
            if (DiG_ptr->adj.count(v_id)) {
                for(auto& p : DiG_ptr->adj.at(v_id)) common_candidates.push_back(p.first);
            }
            if (DiG_ptr->pred.count(v_id)) {
                for(auto& p : DiG_ptr->pred.at(v_id)) common_candidates.push_back(p.first);
            }
            std::sort(common_candidates.begin(), common_candidates.end());
            common_candidates.erase(std::unique(common_candidates.begin(), common_candidates.end()), common_candidates.end());

            // Loop A: Out-neighbors
            if (DiG_ptr->adj.count(v_id)) {
                for (const auto& neighbor_info : DiG_ptr->adj.at(v_id)) {
                    node_t u_id = neighbor_info.first;
                    double scale_u_max = (u_id < scale_max_vec.size()) ? scale_max_vec[u_id] : 0;
                    double r_vu = 0;

                    for (const auto& w_id : common_candidates) {
                        if (u_id == w_id) continue;
                        weight_t mw_uw = compute_directed_mutual_weight(*DiG_ptr, u_id, w_id, weight_key);
                        if (mw_uw == 0) continue; 
                        weight_t mw_vw = compute_directed_mutual_weight(*DiG_ptr, v_id, w_id, weight_key);

                        double p_iq = (scale_v_sum > 0) ? (mw_vw / scale_v_sum) : 0;
                        double m_jq = (scale_u_max > 0) ? (mw_uw / scale_u_max) : 0;
                        r_vu += p_iq * m_jq;
                    }
                    redundancy_sum += (1.0 - r_vu);
                }
            }

            // Loop B: In-neighbors
            if (DiG_ptr->pred.count(v_id)) {
                for (const auto& neighbor_info : DiG_ptr->pred.at(v_id)) {
                    node_t u_id = neighbor_info.first;
                    double scale_u_max = (u_id < scale_max_vec.size()) ? scale_max_vec[u_id] : 0;
                    double r_vu = 0;

                    for (const auto& w_id : common_candidates) {
                        if (u_id == w_id) continue;
                        weight_t mw_uw = compute_directed_mutual_weight(*DiG_ptr, u_id, w_id, weight_key);
                        if (mw_uw == 0) continue; 
                        weight_t mw_vw = compute_directed_mutual_weight(*DiG_ptr, v_id, w_id, weight_key);

                        double p_iq = (scale_v_sum > 0) ? (mw_vw / scale_v_sum) : 0;
                        double m_jq = (scale_u_max > 0) ? (mw_uw / scale_u_max) : 0;
                        r_vu += p_iq * m_jq;
                    }
                    redundancy_sum += (1.0 - r_vu);
                }
            }
            results[i] = redundancy_sum;
        }
    }

    return results;
}

py::object invoke_cpp_effective_size(py::object G, py::object nodes, py::object weight) {
    std::string weight_key = weight.is_none() ? "" : weight.cast<std::string>();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    
    if (nodes.is_none()) nodes = G.attr("nodes");
    py::list nodes_list = py::list(nodes);
    int len = py::len(nodes_list);
    std::vector<node_t> target_ids(len);

    if (py::hasattr(G, "node_to_id")) {
        py::object node_to_id = G.attr("node_to_id"); 
        for (int i = 0; i < len; i++) {
            target_ids[i] = node_to_id[nodes_list[i]].cast<node_t>();
        }
    } else {
        for (int i = 0; i < len; i++) {
            target_ids[i] = nodes_list[i].cast<node_t>();
        }
    }

    std::vector<double> results = compute_redundancy_core(G, target_ids, weight_key, is_directed);
    
    py::array::ShapeContainer ret_shape{ (long)results.size() };
    return py::array_t<double>(ret_shape, results.data());
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

#ifdef EASYGRAPH_ENABLE_GPU
static py::object invoke_gpu_efficiency(py::object G, py::object nodes, py::object weight) {
    Graph& G_ = G.cast<Graph&>();
    py::dict effective_size = py::dict();
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

    py::dict effective_size_dict;
    for (auto node : nodes_list) {
        int node_id = G_.node_to_id[node].cast<node_t>();
        int idx = node2idx[node_id];

        py::object node_name = G_.id_to_node.attr("get")(py::cast(node_id));
        effective_size_dict[node_name] = py::cast(effective_size_results[idx]);
    }
        py::dict degree;
    if (weight.is_none()) {
        degree = G.attr("degree")(py::none()).cast<py::dict>();
    } else {
        degree = G.attr("degree")(weight).cast<py::dict>();
    }

    py::dict efficiency_dict;
    for (auto item : effective_size_dict) {
        int node = py::reinterpret_borrow<py::int_>(item.first).cast<int>();
        double eff_size = py::reinterpret_borrow<py::float_>(item.second).cast<double>();

        if (!degree.contains(py::cast(node))) {
            continue;
        }

        double node_degree = py::reinterpret_borrow<py::float_>(degree[py::cast(node)]).cast<double>();
        if (node_degree == 0.0) {
            efficiency_dict[py::cast(node)] = py::cast(Py_NAN);
        } else {
            double efficiency_value = eff_size / node_degree;
            efficiency_dict[py::cast(node)] = py::cast(efficiency_value);
        }
    }

    return efficiency_dict;
}
#endif


py::object invoke_cpp_efficiency(py::object G, py::object nodes, py::object weight, py::object n_workers) {
    std::string weight_key = weight.is_none() ? "" : weight.cast<std::string>();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    
    // Parsing Nodes
    if (nodes.is_none()) nodes = G.attr("nodes");
    py::list nodes_list = py::list(nodes);
    int len = py::len(nodes_list);
    std::vector<node_t> target_ids(len);

    if (py::hasattr(G, "node_to_id")) {
        py::object node_to_id = G.attr("node_to_id"); 
        for (int i = 0; i < len; i++) {
            target_ids[i] = node_to_id[nodes_list[i]].cast<node_t>();
        }
    } else {
        for (int i = 0; i < len; i++) {
            target_ids[i] = nodes_list[i].cast<node_t>();
        }
    }

    // Compute Efficiency = Effective Size / Degree
    std::vector<double> eff_sizes = compute_redundancy_core(G, target_ids, weight_key, is_directed);

    // Cast Graph pointers for fast degree access
    const Graph* G_ptr = nullptr;
    const DiGraph* DiG_ptr = nullptr;
    if (is_directed) DiG_ptr = &G.cast<const DiGraph&>();
    else G_ptr = &G.cast<const Graph&>();

    std::vector<double> efficiency_results(len);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < len; ++i) {
        double es = eff_sizes[i];
        
        // Propagate NAN from core
        if (std::isnan(es)) {
            efficiency_results[i] = NAN;
            continue;
        }

        node_t v = target_ids[i];
        double degree = 0;

        if (is_directed) {
            if (DiG_ptr->adj.count(v)) degree += DiG_ptr->adj.at(v).size();
            if (DiG_ptr->pred.count(v)) degree += DiG_ptr->pred.at(v).size();
        } else {
            if (G_ptr->adj.count(v)) degree += G_ptr->adj.at(v).size();
        }

        if (degree > 0) {
            efficiency_results[i] = es / degree;
        } else {
            efficiency_results[i] = NAN; 
        }
    }

    py::array::ShapeContainer ret_shape{ (long)len };
    return py::array_t<double>(ret_shape, efficiency_results.data());
}

py::object efficiency(py::object G, py::object nodes, py::object weight, py::object n_workers) {
#ifdef EASYGRAPH_ENABLE_GPU
    return invoke_gpu_efficiency(G, nodes, weight);
#else
    return invoke_cpp_efficiency(G, nodes, weight, n_workers);
#endif
}

py::object invoke_cpp_hierarchy(py::object G, py::object nodes, py::object weight, py::object n_workers) {
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

    std::vector<double> hierarchy_results(nodes_list_len, 0.0);

    // Release GIL for parallel computation
    {
        py::gil_scoped_release release;
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nodes_list_len; i++) {
            node_t u = node_ids[i];
            
            // Validate node strength
            auto str_it = strength.find(u);
            if (str_it == strength.end() || str_it->second == 0.0) continue;
            double u_strength = str_it->second;

            auto& neighbors_u = weighted_adj[u];
            int N = neighbors_u.size();
            if (N <= 1) continue; 

            // Calculate dyadic constraint components
            std::unordered_map<node_t, double> contrib; 

            // Direct 
            for (auto& neighbor : neighbors_u) {
                node_t j = neighbor.first;
                double p_uj = neighbor.second / u_strength;
                contrib[j] += p_uj;
            }

            // Indirect
            for (auto& neighbor_j : neighbors_u) {
                node_t q = neighbor_j.first; 
                double p_uq = neighbor_j.second / u_strength;

                auto q_it = weighted_adj.find(q);
                if (q_it != weighted_adj.end()) {
                    double q_strength = strength[q];
                    for (auto& neighbor_k : q_it->second) {
                        node_t j = neighbor_k.first;
                        if (j == u) continue;

                        // Check if closed triad exists
                        if (weighted_adj[u].count(j)) {
                            double p_qj = neighbor_k.second / q_strength;
                            contrib[j] += p_uq * p_qj;
                        }
                    }
                }
            }

            // Compute Hierarchy score
            double C_total = 0.0;
            std::unordered_map<node_t, double> C_j; 

            for (auto& neighbor : neighbors_u) {
                node_t j = neighbor.first;
                if (contrib.count(j)) {
                    double val = std::pow(contrib[j], 2); 
                    C_j[j] = val;
                    C_total += val;
                }
            }

            if (C_total > 0) {
                double hierarchy_sum = 0.0;
                double denominator = N * std::log(N);

                for (auto& item : C_j) {
                    double c_val = item.second;
                    if (c_val > 0) {
                        double p_i = c_val / C_total;
                        double term = p_i * N;
                        if (term > 0) {
                            hierarchy_sum += term * std::log(term);
                        }
                    }
                }
                hierarchy_results[i] = hierarchy_sum / denominator;
            }
        }
    }

    py::array::ShapeContainer ret_shape{nodes_list_len};
    py::array_t<double> ret(ret_shape, hierarchy_results.data());
    return ret;
}

#ifdef EASYGRAPH_ENABLE_GPU
static py::object invoke_gpu_hierarchy(py::object G, py::object nodes, py::object weight) {
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
    std::vector<double> hierarchy_results;
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

    int gpu_r = gpu_easygraph::hierarchy(V, E, row, col, num_nodes, *W_p, is_directed, node_mask, hierarchy_results);
    if (gpu_r != gpu_easygraph::EG_GPU_SUCC) {
        py::pybind11_fail(gpu_easygraph::err_code_detail(gpu_r));
    }
    py::dict hierarchy_dict;
    for (auto node : nodes_list) {
        int node_id = G_.node_to_id[node].cast<node_t>();
        int idx = node2idx[node_id];

        py::object node_name = G_.id_to_node.attr("get")(py::cast(node_id));
        hierarchy_dict[node_name] = py::cast(hierarchy_results[idx]);
    }
    return hierarchy_dict;
}
#endif

py::object hierarchy(py::object G, py::object nodes, py::object weight, py::object n_workers) {
#ifdef EASYGRAPH_ENABLE_GPU
    return invoke_gpu_hierarchy(G, nodes, weight);
#else
    return invoke_cpp_hierarchy(G, nodes, weight, n_workers);
#endif
}