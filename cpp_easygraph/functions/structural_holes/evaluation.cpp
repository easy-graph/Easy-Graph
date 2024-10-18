#include "evaluation.h"
#include <iomanip>  // 需要引入该头文件来使用 std::setprecision

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
    weight_t nmw;
    if (nmw_rec.count(edge)) {
        nmw = nmw_rec[edge];
    } else {
        weight_t scale = 0;
        for (auto& w : G.adj[u]) {
            weight_t temp_weight = mutual_weight(G, u, w.first, weight);
            scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
        }
        nmw = scale ? (mutual_weight(G, u, v, weight) / scale) : 0;
        nmw_rec[edge] = nmw;
    }
    return nmw;
}

weight_t directed_normalized_mutual_weight(DiGraph& G, node_t u, node_t v, std::string weight, norm_t norm, rec_type& nmw_rec) {
    std::pair<node_t, node_t> edge = std::make_pair(u, v);
    weight_t nmw;
    if (nmw_rec.count(edge)) {
        nmw = nmw_rec[edge];
    } else {
        weight_t scale = 0;
        for (auto& w : G.adj[u]) {
            weight_t temp_weight = directed_mutual_weight(G, u, w.first, weight);
            scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
        }
        for (auto& w : G.pred[u]) {
            weight_t temp_weight = directed_mutual_weight(G, u, w.first, weight);
            scale = (norm == sum) ? (scale + temp_weight) : std::max(scale, temp_weight);
        }
        nmw = scale ? (directed_mutual_weight(G, u, v, weight) / scale) : 0;
        nmw_rec[edge] = nmw;
    }
    return nmw;
}

weight_t directed_local_constraint(DiGraph& G, node_t u, node_t v, std::string weight, rec_type& local_constraint_rec, rec_type& sum_nmw_rec) {
    std::pair<node_t, node_t> edge = std::make_pair(u, v);
    if (local_constraint_rec.count(edge)) {
        return local_constraint_rec[edge];
    } else {
        weight_t direct = directed_normalized_mutual_weight(G, u, v, weight, sum, sum_nmw_rec);
        weight_t indirect = 0;
        std::unordered_set<node_t> neighbors;
        for (const auto& n : G.adj[v]) {
            neighbors.insert(n.first);
        }
        for (const auto& n : G.pred[v]) {
            neighbors.insert(n.first);
        }
        for (const auto& n : neighbors) {
            if (n == v) {
                continue;
            }
            indirect += directed_normalized_mutual_weight(G, u, n, weight, sum, sum_nmw_rec) * 
                        directed_normalized_mutual_weight(G, n, v, weight, sum, sum_nmw_rec);
        }
        weight_t result = pow((direct + indirect), 2);
        local_constraint_rec[edge] = result;
        return result;
    }
}

weight_t local_constraint(Graph& G, node_t u, node_t v, std::string weight, rec_type& local_constraint_rec, rec_type& sum_nmw_rec) {
    std::pair<node_t, node_t> edge = std::make_pair(u, v);
    if (local_constraint_rec.count(edge)) {
        return local_constraint_rec[edge];
    } else {
        weight_t direct = normalized_mutual_weight(G, u, v, weight, sum, sum_nmw_rec);
        weight_t indirect = 0;
        for (auto& w : G.adj[u]) {
            if (w.first == v) {
                continue;
            }
            indirect += normalized_mutual_weight(G, u, w.first, weight, sum, sum_nmw_rec) * 
                        normalized_mutual_weight(G, w.first, v, weight, sum, sum_nmw_rec);
        }
        weight_t result = pow((direct + indirect), 2);
        local_constraint_rec[edge] = result;
        return result;
    }
}

std::pair<node_t, weight_t> compute_constraint_of_v(Graph& G, node_t v, std::string weight, rec_type& local_constraint_rec, rec_type& sum_nmw_rec) {
    weight_t constraint_of_v = 0;
    if (G.adj[v].size() == 0) {
        constraint_of_v = Py_NAN;
    } else {
        for (const auto& n : G.adj[v]) {
            weight_t local_cons = local_constraint(G, v, n.first, weight, local_constraint_rec, sum_nmw_rec);
            constraint_of_v += local_cons;
        }
    }
    return std::make_pair(v, constraint_of_v);
}

std::pair<node_t, weight_t> directed_compute_constraint_of_v(DiGraph& G, node_t v, std::string weight, rec_type& local_constraint_rec, rec_type& sum_nmw_rec) {
    weight_t constraint_of_v = 0;
    if (G.adj[v].size() == 0) {
        constraint_of_v = Py_NAN;
    } else {
        std::unordered_set<node_t> neighbors;
        for (const auto& n : G.adj[v]) {
            neighbors.insert(n.first);
        }
        for (const auto& n : G.pred[v]) {
            neighbors.insert(n.first);
        }
        for (const auto& n : neighbors) {
            weight_t local_cons = directed_local_constraint(G, v, n, weight, local_constraint_rec, sum_nmw_rec);
            constraint_of_v += local_cons;
        }
    }
    return std::make_pair(v, constraint_of_v);
}

py::object invoke_cpp_constraint(py::object G, py::object nodes, py::object weight) {
    std::string weight_key = weight_to_string(weight);
    rec_type sum_nmw_rec, local_constraint_rec;
    if (nodes.is_none()) {
        nodes = G.attr("nodes");
    }
    py::list nodes_list = py::list(nodes);
    py::list constraint_results = py::list();
    int nodes_list_len = py::len(nodes_list);
    if(G.attr("is_directed")().cast<bool>()){
        DiGraph& G_ = G.cast<DiGraph&>();
        for (int i = 0; i < nodes_list_len; i++) {
            py::object v = nodes_list[i];
            node_t v_id = G_.node_to_id[v].cast<node_t>();
            std::pair<node_t, weight_t> constraint_pair = directed_compute_constraint_of_v(G_, v_id, weight_key, local_constraint_rec, sum_nmw_rec);
            py::tuple constraint_of_v = py::make_tuple(G_.id_to_node[py::cast(constraint_pair.first)], constraint_pair.second);
            constraint_results.append(constraint_of_v);
        }
    }else{
        Graph& G_ = G.cast<Graph&>();
        for (int i = 0; i < nodes_list_len; i++) {
            py::object v = nodes_list[i];
            node_t v_id = G_.node_to_id[v].cast<node_t>();
            std::pair<node_t, weight_t> constraint_pair = compute_constraint_of_v(G_, v_id, weight_key, local_constraint_rec, sum_nmw_rec);
            py::tuple constraint_of_v = py::make_tuple(G_.id_to_node[py::cast(constraint_pair.first)], constraint_pair.second);
            constraint_results.append(constraint_of_v);
        }
    }
    py::dict constraint = py::dict(constraint_results);
    return constraint;
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
    std::vector<int>& E = csr_graph->E;
    std::vector<int>& V = csr_graph->V;

    std::vector<double> *W_p = weight.is_none() ? &(csr_graph->unweighted_W) 
                                : csr_graph->W_map.find(weight_to_string(weight))->second.get();
    std::unordered_map<node_t, int>& node2idx = csr_graph->node2idx;

    
    std::vector<double> constraint_results;
    int gpu_r = gpu_easygraph::constraint(V, E, *W_p, constraint_results);
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


weight_t redundancy(DiGraph& G, node_t u, node_t v, std::string weight, rec_type& sum_nmw_rec, rec_type& max_nmw_rec) {
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

py::object effective_size(py::object G, py::object nodes, py::object weight, py::object n_workers) {
    // Graph& G_ = G.cast<Graph&>();
    rec_type sum_nmw_rec, max_nmw_rec;
    py::dict effective_size = py::dict();
    if (nodes.is_none()) {
        nodes = G;
    }
    nodes = py::list(nodes);
    if (!G.attr("is_directed")().cast<bool>() && weight.is_none()) {
        int nodes_len = py::len(nodes);
        for (int i = 0; i < nodes_len; i++) {
            py::object v = nodes[py::cast(i)];
            if (py::len(G[v]) == 0) {
                effective_size[v] = py::cast(Py_NAN);
                continue;
            }
            py::object E = G.attr("ego_subgraph")(v);
            E.attr("remove_node")(v);
            weight_t size = E.attr("size")().cast<weight_t>();
            effective_size[v] = py::len(E) - (2 * size) / py::len(E);
        }
    } else {
        DiGraph& G_ = G.cast<DiGraph&>();
        std::string weight_key = weight_to_string(weight);
        int nodes_len = py::len(nodes);
        for (int i = 0; i < nodes_len; i++) {
            py::object v = nodes[py::cast(i)];
            if (py::len(G[v]) == 0) {
                effective_size[v] = py::cast(Py_NAN);
                continue;
            }
            weight_t redundancy_sum = 0;
            node_t v_id = G_.node_to_id[v].cast<node_t>();
            std::unordered_set<node_t> neighbors;
            for (const auto& n : G_.adj[v_id]) {
                neighbors.insert(n.first);
            }
            for (const auto& n : G_.pred[v_id]) {
                neighbors.insert(n.first);
            }
            for (const auto& u_id : neighbors) {
                redundancy_sum += redundancy(G_, v_id, u_id, weight_key, sum_nmw_rec, max_nmw_rec);
            }
            effective_size[v] = redundancy_sum;
        }
    }
    return effective_size;
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

py::object hierarchy(py::object G, py::object nodes, py::object weight, py::object n_workers) {
    rec_type local_constraint_rec, sum_nmw_rec;
    std::string weight_key = weight_to_string(weight);
    if (nodes.is_none()) {
        nodes = G.attr("nodes");
    }
    py::list nodes_list = py::list(nodes);

    Graph& G_ = G.cast<Graph&>();
    py::dict hierarchy = py::dict();
    int nodes_list_len = py::len(nodes_list);
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
    return hierarchy;
}