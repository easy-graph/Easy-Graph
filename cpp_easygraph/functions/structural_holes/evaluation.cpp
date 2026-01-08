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

weight_t mutual_weight(Graph& G, node_t u, node_t v, std::string weight) {
    weight_t a_uv = 0, a_vu = 0;
    if (G.adj.count(u) && G.adj[u].count(v)) {
        edge_attr_dict_factory& guv = G.adj[u][v];
        a_uv = guv.count(weight) ? guv[weight] : 1;
    }
    if (G.adj.count(v) && G.adj[v].count(u)) {
        edge_attr_dict_factory& gvu = G.adj[v][u];
        a_vu = gvu.count(weight) ? gvu[weight] : 1;
    }
    return a_uv + a_vu;
}

weight_t directed_mutual_weight(DiGraph& G, node_t u, node_t v, std::string weight) {
    weight_t a_uv = 0, a_vu = 0;
    if (G.adj.count(u) && G.adj[u].count(v)) {
        edge_attr_dict_factory& guv = G.adj[u][v];
        a_uv = guv.count(weight) ? guv[weight] : 1;
    }
    if (G.adj.count(v) && G.adj[v].count(u)) {
        edge_attr_dict_factory& gvu = G.adj[v][u];
        a_vu = gvu.count(weight) ? gvu[weight] : 1;
    }
    return a_uv + a_vu;
}

weight_t normalized_mutual_weight(Graph& G, node_t u, node_t v, std::string weight, norm_t norm, rec_type& nmw_rec) {
    std::pair<node_t, node_t> edge = std::make_pair(u, v);
    if (nmw_rec.count(edge)) return nmw_rec[edge];
    
    weight_t scale = 0;
    for (auto& w : G.adj[u]) {
        weight_t temp_weight = mutual_weight(G, u, w.first, weight);
        scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
    }
    weight_t nmw = scale ? (mutual_weight(G, u, v, weight) / scale) : 0;
    nmw_rec[edge] = nmw;
    return nmw;
}

weight_t directed_normalized_mutual_weight(DiGraph& G, node_t u, node_t v, std::string weight, norm_t norm, rec_type& nmw_rec) {
    std::pair<node_t, node_t> edge = std::make_pair(u, v);
    if (nmw_rec.count(edge)) return nmw_rec[edge];

    weight_t scale = 0;
    for (auto& w : G.adj[u]) {
        weight_t temp_weight = directed_mutual_weight(G, u, w.first, weight);
        scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
    }
    for (auto& w : G.pred[u]) {
        weight_t temp_weight = directed_mutual_weight(G, u, w.first, weight);
        scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
    }
    weight_t nmw = scale ? (directed_mutual_weight(G, u, v, weight) / scale) : 0;
    nmw_rec[edge] = nmw;
    return nmw;
}

weight_t local_constraint(Graph& G, node_t u, node_t v, std::string weight, rec_type& local_constraint_rec, rec_type& sum_nmw_rec) {
    std::pair<node_t, node_t> edge = std::make_pair(u, v);
    if (local_constraint_rec.count(edge)) return local_constraint_rec[edge];

    weight_t direct = normalized_mutual_weight(G, u, v, weight, sum, sum_nmw_rec);
    weight_t indirect = 0;
    for (auto& w : G.adj[u]) {
        if (w.first == v) continue;
        indirect += normalized_mutual_weight(G, u, w.first, weight, sum, sum_nmw_rec) *
                    normalized_mutual_weight(G, w.first, v, weight, sum, sum_nmw_rec);
    }
    weight_t result = pow((direct + indirect), 2);
    local_constraint_rec[edge] = result;
    return result;
}

weight_t directed_local_constraint(DiGraph& G, node_t u, node_t v, std::string weight, rec_type& local_constraint_rec, rec_type& sum_nmw_rec) {
    std::pair<node_t, node_t> edge = std::make_pair(u, v);
    if (local_constraint_rec.count(edge)) return local_constraint_rec[edge];

    weight_t direct = directed_normalized_mutual_weight(G, u, v, weight, sum, sum_nmw_rec);
    weight_t indirect = 0;
    std::unordered_set<node_t> neighbors;
    for (const auto& n : G.adj[v]) neighbors.insert(n.first);
    for (const auto& n : G.pred[v]) neighbors.insert(n.first);
    
    for (const auto& n : neighbors) {
        if (n == v) continue;
        indirect += directed_normalized_mutual_weight(G, u, n, weight, sum, sum_nmw_rec) *
                    directed_normalized_mutual_weight(G, n, v, weight, sum, sum_nmw_rec);
    }
    weight_t result = pow((direct + indirect), 2);
    local_constraint_rec[edge] = result;
    return result;
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

weight_t redundancy(Graph& G, node_t u, node_t v, std::string weight, rec_type& sum_nmw_rec, rec_type& max_nmw_rec) {
    weight_t r = 0;
    std::unordered_set<node_t> neighbors;
    for (const auto& n : G.adj[v]) {
        neighbors.insert(n.first);
    }
    for (const auto& w : neighbors) {
        r += normalized_mutual_weight(G, u, w, weight, sum, sum_nmw_rec) * normalized_mutual_weight(G, v, w, weight, max, max_nmw_rec);
    }
    return 1 - r;
}

weight_t directed_redundancy(DiGraph& G, node_t u, node_t v, std::string weight, rec_type& sum_nmw_rec, rec_type& max_nmw_rec) {
    weight_t r = 0;
    std::unordered_set<node_t> neighbors;
    for (const auto& n : G.adj[v]) {
        neighbors.insert(n.first);
    }
    for (const auto& n : G.pred[v]) {
        neighbors.insert(n.first);
    }
    for (const auto& w : neighbors) {
        r += directed_normalized_mutual_weight(G, u, w, weight, sum, sum_nmw_rec) * directed_normalized_mutual_weight(G, v, w, weight, max, max_nmw_rec);
    }
    return 1 - r;
}

py::object invoke_cpp_effective_size(py::object G, py::object nodes, py::object weight) {
    std::string weight_key = weight_to_string(weight);
    rec_type sum_nmw_rec, max_nmw_rec;

    if (nodes.is_none()) {
        nodes = G.attr("nodes");
    }

    py::list nodes_list = py::list(nodes);
    int nodes_list_len = py::len(nodes_list);
    std::vector<double> effective_size_results(nodes_list_len, 0.0);

    if (!G.attr("is_directed")().cast<bool>()){
        Graph& G_ = G.cast<Graph&>();
        for (int i = 0; i < nodes_list_len; i++) {
            weight_t redundancy_sum = 0;
            py::object v = nodes_list[i];
            node_t v_id = G_.node_to_id[v].cast<node_t>();
            if (py::len(G[v]) == 0) {
                effective_size_results[i] = Py_NAN;
                continue;
            }
            for (const auto& neighbor_info : G_.adj[v_id]) {
                node_t u_id = neighbor_info.first;
                redundancy_sum += redundancy(G_, v_id, u_id, weight_key, sum_nmw_rec, max_nmw_rec);
            }
            effective_size_results[i] = redundancy_sum;
        }
    } else{
        DiGraph& G_ = G.cast<DiGraph&>();
        for (int i = 0; i < nodes_list_len; i++) {
            weight_t redundancy_sum = 0;
            py::object v = nodes_list[i];
            node_t v_id = G_.node_to_id[v].cast<node_t>();
            if (py::len(G[v]) == 0) {
                effective_size_results[i] = Py_NAN;
                continue;
            }
            for (const auto& neighbor_info : G_.adj[v_id]) {
                node_t u_id = neighbor_info.first;
                redundancy_sum += directed_redundancy(G_, v_id, u_id, weight_key, sum_nmw_rec, max_nmw_rec);
            }
            for (const auto& neighbor_info : G_.pred[v_id]) {
                node_t u_id = neighbor_info.first;
                redundancy_sum += directed_redundancy(G_, v_id, u_id, weight_key, sum_nmw_rec, max_nmw_rec);
            }
            effective_size_results[i] = redundancy_sum;
        }
    }

    std::reverse(effective_size_results.begin(), effective_size_results.end());

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
    rec_type sum_nmw_rec, max_nmw_rec;
    py::dict effective_size_dict = py::dict();
    if (nodes.is_none()) {
        nodes = G;
    }
    nodes = py::list(nodes);
    if (!G.attr("is_directed")().cast<bool>()){
        Graph& G_ = G.cast<Graph&>();
        std::string weight_key = weight_to_string(weight);
        int nodes_len = py::len(nodes);
        for (int i = 0; i < nodes_len; i++) {
            py::object v = nodes[py::cast(i)];
            if (py::len(G[v]) == 0) {
                effective_size_dict[v] = py::cast(Py_NAN);
                continue;
            }
            weight_t redundancy_sum = 0;
            node_t v_id = G_.node_to_id[v].cast<node_t>();
            for (const auto& neighbor_info : G_.adj[v_id]) {
                node_t u_id = neighbor_info.first;
                redundancy_sum += redundancy(G_, v_id, u_id, weight_key, sum_nmw_rec, max_nmw_rec);
            }
            effective_size_dict[v] = redundancy_sum;
        }
    } else{
        DiGraph& G_ = G.cast<DiGraph&>();
        std::string weight_key = weight_to_string(weight);
        int nodes_len = py::len(nodes);
        for (int i = 0; i < nodes_len; i++) {
            py::object v = nodes[py::cast(i)];
            if (py::len(G[v]) == 0) {
                effective_size_dict[v] = py::cast(Py_NAN);
                continue;
            }
            weight_t redundancy_sum = 0;
            node_t v_id = G_.node_to_id[v].cast<node_t>();
            for (const auto& neighbor_info : G_.adj[v_id]) {
                node_t u_id = neighbor_info.first;
                redundancy_sum += directed_redundancy(G_, v_id, u_id, weight_key, sum_nmw_rec, max_nmw_rec);
            }
            for (const auto& neighbor_info : G_.pred[v_id]) {
                node_t u_id = neighbor_info.first;
                redundancy_sum += directed_redundancy(G_, v_id, u_id, weight_key, sum_nmw_rec, max_nmw_rec);
            }
            effective_size_dict[v] = redundancy_sum;
        }
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

py::object efficiency(py::object G, py::object nodes, py::object weight, py::object n_workers) {
#ifdef EASYGRAPH_ENABLE_GPU
    return invoke_gpu_efficiency(G, nodes, weight);
#else
    return invoke_cpp_efficiency(G, nodes, weight, n_workers);
#endif
}

void hierarchy_parallel(Graph* G, std::vector<node_t>* nodes, std::string weight, std::unordered_map<node_t, weight_t>* ret) {
    rec_type local_constraint_rec, sum_nmw_rec;
    for (node_t v : *nodes) {
        int n = G->adj[v].size(); // len(G.ego_subgraph(v)) - 1
        weight_t C = 0;
        std::unordered_map<node_t, weight_t> c;
        for (const auto& w_pair : G->adj[v]) {
            node_t w = w_pair.first;
            C += local_constraint(*G, v, w, weight, local_constraint_rec, sum_nmw_rec);
            c[w] = local_constraint(*G, v, w, weight, local_constraint_rec, sum_nmw_rec);
        }
        if (n > 1) {
            weight_t sum = 0;
            for (const auto& w_pair : G->adj[v]) {
                node_t w = w_pair.first;
                sum += c[w] / C * n * log(c[w] / C * n) / (n * log(n));
            }
            (*ret)[v] = sum;
        }
        else {
            (*ret)[v] = 0;
        }
    }
}

inline std::vector<std::vector<node_t> > split_len(const std::vector<node_t>& nodes, int step) {
    std::vector<std::vector<node_t> > ret;
    for (int i = 0; i < nodes.size();i += step) {
        ret.emplace_back(nodes.begin() + i, (i + step > nodes.size()) ? nodes.end() : nodes.begin() + i + step);
    }
    if (ret.back().size() * 3 < step) {
        ret[ret.size() - 2].insert(ret[ret.size() - 2].end(), ret.back().begin(), ret.back().end());
        ret.pop_back();
    }
    return ret;
}

inline std::vector<std::vector<node_t> > split(const std::vector<node_t>& nodes, int n) {
    std::vector<std::vector<node_t> > ret;
    int length = nodes.size();
    int step = length / n + 1;
    for (int i = 0;i < length;i += step) {
        ret.emplace_back(nodes.begin() + i, i + step > length ? nodes.end() : nodes.begin() + i + step);
    }
    return ret;
}

py::object invoke_cpp_hierarchy(py::object G, py::object nodes, py::object weight, py::object n_workers) {
    rec_type local_constraint_rec, sum_nmw_rec;
    std::string weight_key = weight_to_string(weight);
    if (nodes.is_none()) {
        nodes = G.attr("nodes");
    }
    py::list nodes_list = py::list(nodes);
    int nodes_list_len = py::len(nodes_list);
    py::dict hierarchy = py::dict();

    if(G.attr("is_directed")().cast<bool>()){
        DiGraph& G_ = G.cast<DiGraph&>();
        for (int i = 0; i < nodes_list_len; i++) {
            py::object v = nodes_list[i];
            weight_t C = 0;
            std::map<node_t, weight_t> c;

            py::list successors_of_v = py::list(G.attr("successors")(v));
            py::list predecessors_of_v = py::list(G.attr("predecessors")(v));

            std::set<node_t> neighbors_of_v;
            for (const auto& w : successors_of_v) {
                neighbors_of_v.insert(G_.node_to_id[w].cast<node_t>());
            }
            for (const auto& w : predecessors_of_v) {
                neighbors_of_v.insert(G_.node_to_id[w].cast<node_t>());
            }

            for (const auto& w_id : neighbors_of_v) {
                node_t v_id = G_.node_to_id[v].cast<node_t>();

                C += directed_local_constraint(G_, v_id, w_id, weight_key, local_constraint_rec, sum_nmw_rec);
                c[w_id] = directed_local_constraint(G_, v_id, w_id, weight_key, local_constraint_rec, sum_nmw_rec);
            }
            int n = neighbors_of_v.size();

            if (n > 1) {
                weight_t hierarchy_sum = 0;
                for (const auto& w_id : neighbors_of_v) {
                    hierarchy_sum += c[w_id] / C * n * log(c[w_id] / C * n) / (n * log(n));
                }
                hierarchy[v] = hierarchy_sum;
            }

            if (!hierarchy.contains(v)) {
                hierarchy[v] = 0;
            }
        }

    }else{
        Graph& G_ = G.cast<Graph&>();
        if (!n_workers.is_none()) {
            std::vector<node_t> node_ids;
            int n_workers_num = n_workers.cast<unsigned>();
            for (int i = 0;i < py::len(nodes_list);i++) {
                py::object node = nodes_list[i];
                node_ids.push_back(G_.node_to_id[node].cast<node_t>());
            }
            std::shuffle(node_ids.begin(), node_ids.end(), std::random_device());
            std::vector<std::vector<node_t> > split_nodes;
            if (node_ids.size() > n_workers_num * 30000) {
                split_nodes = split_len(node_ids, 30000);
            }
            else {
                split_nodes = split(node_ids, n_workers_num);
            }
            while (split_nodes.size() < n_workers_num) {
                split_nodes.push_back(std::vector<node_t>());
            }
            std::vector<std::unordered_map<node_t, weight_t> > rets(n_workers_num);
            Py_BEGIN_ALLOW_THREADS

                std::vector<std::thread> threads;
                for (int i = 0;i < n_workers_num; i++) {
                    threads.push_back(std::thread(hierarchy_parallel, &G_, &split_nodes[i], weight_key, &rets[i]));
                }
                for (int i = 0;i < n_workers_num;i++) {
                    threads[i].join();
                }

            Py_END_ALLOW_THREADS

            for (int i = 1;i < rets.size();i++) {
                rets[0].insert(rets[i].begin(), rets[i].end());
            }
            for (const auto& hierarchy_pair : rets[0]) {
                py::object node = G_.id_to_node[py::cast(hierarchy_pair.first)];
                hierarchy[node] = hierarchy_pair.second;
            }
        }
        else {
            for (int i = 0; i < nodes_list_len; i++) {
                py::object v = nodes_list[i];
                py::object E = G.attr("ego_subgraph")(v);

                int n = py::len(E) - 1;

                weight_t C = 0;
                std::map<node_t, weight_t> c;
                py::list neighbors_of_v = py::list(G.attr("neighbors")(v));
                int neighbors_of_v_len = py::len(neighbors_of_v);
                for (int j = 0; j < neighbors_of_v_len; j++) {
                    py::object w = neighbors_of_v[j];
                    node_t v_id = G_.node_to_id[v].cast<node_t>();
                    node_t w_id = G_.node_to_id[w].cast<node_t>();
                    C += local_constraint(G_, v_id, w_id, weight_key, local_constraint_rec, sum_nmw_rec);
                    c[w_id] = local_constraint(G_, v_id, w_id, weight_key, local_constraint_rec, sum_nmw_rec);
                }
                if (n > 1) {
                    weight_t hierarchy_sum = 0;
                    int neighbors_of_v_len = py::len(neighbors_of_v);
                    for (int k = 0; k < neighbors_of_v_len; k++) {
                        py::object w = neighbors_of_v[k];
                        node_t w_id = G_.node_to_id[w].cast<node_t>();
                        hierarchy_sum += c[w_id] / C * n * log(c[w_id] / C * n) / (n * log(n));
                    }
                    hierarchy[v] = hierarchy_sum;
                }
                if (!hierarchy.contains(v)) {
                    hierarchy[v] = 0;
                }
            }
        }
    }


    return hierarchy;
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