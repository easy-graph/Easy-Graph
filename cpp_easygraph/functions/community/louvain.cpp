#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef _OPENMP
#include <omp.h>
#else
#warning "OpenMP is not available: cpp_louvain_communities will fall back to single-threaded execution."
#endif
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include "../../functions/community/graph_coloring.h"

namespace py = pybind11;
using namespace std;

const double epsilon = 1e-10;
const int MAX_ITERATIONS = 50;
const double CONVERGENCE_THRESHOLD = 1e-7;

struct CommunityState {
    int size = 0;
    double weight_inside = 0;
    double weight_all = 0;
};

double calculate_norm(const Graph_L& G) {
    double total_weight = 0.0;
    for (int u = 1; u < G.n + 1; ++u) {
        for (int i = G.head[u]; i != -1; i = G.edges[i].next) {
            total_weight += G.edges[i].w;
        }
    }
    return total_weight;
}

double calculate_actual_modularity(const Graph_L& G, const vector<int>& membership,
                                  double norm, double resolution) {
    unordered_map<int, double> comm_weight_inside;
    unordered_map<int, double> comm_weight_total;

    for (int u = 1; u <= G.n; ++u) {
        int comm_u = membership[u - 1];
        for (int e_idx = G.head[u]; e_idx != -1; e_idx = G.edges[e_idx].next) {
            int v = G.edges[e_idx].to;
            double w = G.edges[e_idx].w;
            int comm_v = membership[v - 1];
            comm_weight_total[comm_u] += w;
            if (comm_u == comm_v) {
                comm_weight_inside[comm_u] += w;
            }
        }
    }

    double modularity = 0.0;
    for (const auto& kv : comm_weight_inside) {
        int comm = kv.first;
        double inside = kv.second;
        double total = comm_weight_total[comm];
        modularity += (inside / 2.0) - resolution * (total * total) / (4.0 * norm * norm);
    }

    return modularity;
}

inline double calculate_modularity_gain(double comm_weight_without_u, double norm,
                                        double node_weight_all, double weight_to_comm, double resolution) {
    double gain = weight_to_comm - resolution * (comm_weight_without_u * node_weight_all) / norm;
    return 2 * gain / norm;
}

double modularity_optimization_serial(const Graph_L& G, vector<int>& membership, double resolution_val, double norm) {
    int n = G.n;
    double total_modularity_gain = 0.0;
    vector<CommunityState> comms(n);

    for (int i = 0; i < n; ++i) {
        membership[i] = i;
        comms[i].size = 1;
        comms[i].weight_all = 0.0;
        comms[i].weight_inside = 0.0;
        for (int j = G.head[i + 1]; j != -1; j = G.edges[j].next) {
            int v = G.edges[j].to;
            double w = G.edges[j].w;
            comms[i].weight_all += w;
            if (v == i + 1) comms[i].weight_inside += w;
        }
    }

    vector<double> node_weight(n + 1, 0.0);
    vector<double> node_loop(n + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        node_weight[i + 1] = comms[i].weight_all;
        node_loop[i + 1] = comms[i].weight_inside;
    }

    vector<double> neigh_weight(n, 0.0);
    vector<int> timestamp(n, 0);
    vector<int> touched;
    touched.reserve(512);
    int version = 0;

    std::random_device rd;
    std::mt19937 rng(rd());
    vector<int> node_order(n);
    for (int i = 0; i < n; ++i) node_order[i] = i + 1;

    int num_moved_nodes = 1;
    int iteration = 0;
    double previous_modularity = -1e100;

    do {
        num_moved_nodes = 0;
        iteration++;

        std::shuffle(node_order.begin(), node_order.end(), rng);

        for (int idx = 0; idx < n; ++idx) {
            int u = node_order[idx];

            touched.clear();
            version++;
            if (version == 0) version = 1;

            double weight_all = node_weight[u];
            double weight_loop = node_loop[u];
            double weight_inside = 0.0;
            int my_comm = membership[u - 1];

            for (int e = G.head[u]; e != -1; e = G.edges[e].next) {
                int v = G.edges[e].to;
                double w = G.edges[e].w;

                if (u == v) continue;

                int v_comm = membership[v - 1];

                if (timestamp[v_comm] != version) {
                    timestamp[v_comm] = version;
                    neigh_weight[v_comm] = 0.0;
                    touched.push_back(v_comm);
                }
                neigh_weight[v_comm] += w;

                if (v_comm == my_comm) {
                    weight_inside += w;
                }
            }

            if (weight_all < epsilon) continue;

            int old_comm = my_comm;
            double comm_old_w = comms[old_comm].weight_all - weight_all;
            double old_gain = calculate_modularity_gain(comm_old_w, norm, weight_all, weight_inside, resolution_val);

            double max_gain = std::max(old_gain, 0.0);
            int best_comm = old_comm;
            double best_weight_to_comm = weight_inside;

            for (int target_comm : touched) {
                if (target_comm == old_comm) continue;

                double w_to_target = neigh_weight[target_comm];
                double comm_target_w = comms[target_comm].weight_all;

                double gain = calculate_modularity_gain(comm_target_w, norm, weight_all, w_to_target, resolution_val);

                if (gain > max_gain + epsilon) {
                    max_gain = gain;
                    best_comm = target_comm;
                    best_weight_to_comm = w_to_target;
                }
            }

            if (best_comm != old_comm) {
                comms[old_comm].size--;
                comms[old_comm].weight_all -= weight_all;
                comms[old_comm].weight_inside -= (2.0 * weight_inside + weight_loop);

                comms[best_comm].size++;
                comms[best_comm].weight_all += weight_all;
                comms[best_comm].weight_inside += (2.0 * best_weight_to_comm + weight_loop);

                membership[u - 1] = best_comm;
                num_moved_nodes++;
            }
        }

        if (num_moved_nodes == 0) {
            return calculate_actual_modularity(G, membership, norm, resolution_val);
        }

        if (iteration >= MAX_ITERATIONS) {
            return calculate_actual_modularity(G, membership, norm, resolution_val);
        }

    } while (true);

    for (int i = 0; i < n; ++i) {
        comms[i].weight_all = node_weight[i + 1];
    }

    return calculate_actual_modularity(G, membership, norm, resolution_val);
}

double modularity_optimization_parallel_simplified(const Graph_L& G, vector<int>& membership, double resolution_val, double norm) {
    int n = G.n;
    vector<CommunityState> comms(n);

    for (int i = 0; i < n; ++i) {
        membership[i] = i;
        comms[i].size = 1;
        comms[i].weight_all = 0.0;
        comms[i].weight_inside = 0.0;
        for (int j = G.head[i + 1]; j != -1; j = G.edges[j].next) {
            int v = G.edges[j].to;
            double w = G.edges[j].w;
            comms[i].weight_all += w;
            if (v == i + 1) comms[i].weight_inside += w;
        }
    }

    vector<double> node_weight(n + 1, 0.0);
    vector<double> node_loop(n + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        node_weight[i + 1] = comms[i].weight_all;
        node_loop[i + 1] = comms[i].weight_inside;
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    vector<int> node_order(n);
    iota(node_order.begin(), node_order.end(), 0);

    int num_moved_nodes = 1;
    int iteration = 0;

    vector<double> comm_weights(n, 0.0);
    for (int i = 0; i < n; ++i) {
        comm_weights[i] = comms[i].weight_all;
    }

    do {
        num_moved_nodes = 0;
        iteration++;

        std::shuffle(node_order.begin(), node_order.end(), rng);

        #pragma omp parallel
        {
            #pragma omp for reduction(+:num_moved_nodes) schedule(guided)
            for (int idx = 0; idx < n; ++idx) {
                int u = node_order[idx];

                double weight_all = node_weight[u];
                int my_comm = membership[u - 1];

                if (weight_all < epsilon) continue;

                unordered_map<int, double> neigh_weight;
                neigh_weight.reserve(32);
                vector<int> touched;
                touched.reserve(32);

                double weight_inside = 0.0;

                for (int e = G.head[u]; e != -1; e = G.edges[e].next) {
                    int v = G.edges[e].to;
                    double w = G.edges[e].w;

                    if (u == v) continue;

                    int v_comm = membership[v - 1];

                    auto it = neigh_weight.find(v_comm);
                    if (it == neigh_weight.end()) {
                        touched.push_back(v_comm);
                        neigh_weight[v_comm] = w;
                    } else {
                        it->second += w;
                    }

                    if (v_comm == my_comm) {
                        weight_inside += w;
                    }
                }

                int old_comm = my_comm;
                double comm_old_w = comm_weights[old_comm] - weight_all;
                double old_gain = calculate_modularity_gain(comm_old_w, norm, weight_all, weight_inside, resolution_val);

                double max_gain = std::max(old_gain, 0.0);
                int best_comm = old_comm;

                for (int target_comm : touched) {
                    if (target_comm == old_comm) continue;

                    double w_to_target = neigh_weight.at(target_comm);
                    double comm_target_w = comm_weights[target_comm];

                    double gain = calculate_modularity_gain(comm_target_w, norm, weight_all, w_to_target, resolution_val);

                    if (gain > max_gain + epsilon) {
                        max_gain = gain;
                        best_comm = target_comm;
                    }
                }

                if (best_comm != old_comm) {
                    #pragma omp atomic update
                    comm_weights[old_comm] -= weight_all;

                    #pragma omp atomic update
                    comm_weights[best_comm] += weight_all;

                    membership[u - 1] = best_comm;
                    num_moved_nodes++;
                }
            }
        }

        if (num_moved_nodes == 0) break;
        if (iteration >= MAX_ITERATIONS) break;

    } while (true);

    for (int i = 0; i < n; ++i) {
        comms[i].weight_all = comm_weights[i];
    }

    return calculate_actual_modularity(G, membership, norm, resolution_val);
}

struct SuperEdge {
    int u, v;
    double w;
};

Graph_L* graph_compress(const Graph_L* G, vector<int>& membership) {
    unordered_map<int, int> new_comm_id;
    int new_vcount = 0;
    for (int i = 0; i < G->n; i++) {
        int c = membership[i];
        if (new_comm_id.find(c) == new_comm_id.end()) {
            new_comm_id[c] = new_vcount++;
        }
        membership[i] = new_comm_id[c];
    }

    vector<SuperEdge> edges;

    for (int u = 1; u < G->n + 1; ++u) {
        int su = membership[u-1] + 1;
        for (int e = G->head[u]; e != -1; e = G->edges[e].next) {
            int v = G->edges[e].to;
            double w = G->edges[e].w;
            int sv = membership[v-1] + 1;
            edges.push_back({su, sv, w});
        }
    }

    sort(edges.begin(), edges.end(), [](const SuperEdge& a, const SuperEdge& b) {
        if (a.u != b.u) return a.u < b.u;
        return a.v < b.v;
    });

    Graph_L* super_g = new Graph_L(new_vcount, G->is_directed, G->is_deg);
    if (edges.empty()) return super_g;

    int curr_u = edges[0].u;
    int curr_v = edges[0].v;
    double curr_w = edges[0].w;

    for (size_t i = 1; i < edges.size(); ++i) {
        if (edges[i].u == curr_u && edges[i].v == curr_v) {
            curr_w += edges[i].w;
        } else {
            super_g->add_weighted_edge(curr_u, curr_v, curr_w);
            curr_u = edges[i].u;
            curr_v = edges[i].v;
            curr_w = edges[i].w;
        }
    }
    super_g->add_weighted_edge(curr_u, curr_v, curr_w);
    return super_g;
}

py::object cpp_louvain_communities_serial(py::object G, py::object weight, py::object threshold, py::object resolution) {
    Graph& G_ = G.cast<Graph&>();

    {
        if (G_.is_linkgraph_dirty()) {
            G_._get_linkgraph_structure();
        }
    }

    Graph_L original_GL = G_._get_linkgraph_structure();

    Graph_L* current_GL = &original_GL;
    double threshold_val = threshold.cast<double>();
    double resolution_val = resolution.cast<double>();
    int vcount = current_GL->n;
    if (vcount == 0) return py::list();

    double norm = calculate_norm(*current_GL);
    vector<int> membership(vcount);
    iota(membership.begin(), membership.end(), 0);

    vector<Graph_L*> allocated_graphs;
    int iteration = 0;
    double previous_modularity = -1e100;

    if (norm > 0.0) {
        while (true) {
            iteration++;
            int curr_vcount = current_GL->n;
            vector<int> curr_membership(curr_vcount);

            double current_modularity;
            current_modularity = modularity_optimization_serial(*current_GL, curr_membership, resolution_val, norm);

            double modularity_gain = current_modularity - previous_modularity;

            if (modularity_gain <= threshold_val || iteration > 100) {
                break;
            }

            Graph_L* next_GL = graph_compress(current_GL, curr_membership);
            allocated_graphs.push_back(next_GL);
            current_GL = next_GL;

            for (int i = 0; i < vcount; i++) {
                membership[i] = curr_membership[membership[i]];
            }

            previous_modularity = current_modularity;
        }
    }

    unordered_map<int, vector<node_t>> cpp_groups;
    for (int i = 0; i < vcount; ++i) {
        int comm_id = membership[i];
        node_t node_id = i + 1;
        cpp_groups[comm_id].push_back(node_id);
    }

    py::dict id_to_node = G_.id_to_node;

    py::list final_result;
    for (auto& kv : cpp_groups) {
        py::set py_comm;
        for (node_t node_id : kv.second) {
            py::object node_obj = id_to_node[py::cast(node_id)];
            py_comm.add(node_obj);
        }
        final_result.append(py_comm);
    }

    for (auto g : allocated_graphs) {
        delete g;
    }

    return final_result;
}

py::object cpp_louvain_communities(py::object G, py::object weight, py::object threshold, py::object resolution) {
    Graph& G_ = G.cast<Graph&>();

    #ifdef _OPENMP
    omp_set_num_threads(8);
    #endif

    Graph_L original_GL = G_._get_linkgraph_structure();

    Graph_L* current_GL = &original_GL;
    double threshold_val = threshold.cast<double>();
    double resolution_val = resolution.cast<double>();
    int vcount = current_GL->n;
    if (vcount == 0) return py::list();

    double norm = calculate_norm(*current_GL);
    vector<int> membership(vcount);
    iota(membership.begin(), membership.end(), 0);

    vector<Graph_L*> allocated_graphs;
    int iteration = 0;
    double previous_modularity = 0.0;

    if (norm > 0.0) {
        while (true) {
            iteration++;
            int curr_vcount = current_GL->n;
            vector<int> curr_membership(curr_vcount);
            double current_modularity;

            current_modularity = modularity_optimization_parallel_simplified(*current_GL, curr_membership, resolution_val, norm);

            double modularity_gain = current_modularity - previous_modularity;
            previous_modularity = current_modularity;

            if (modularity_gain <= threshold_val || iteration > 100) {
                break;
            }

            Graph_L* next_GL = graph_compress(current_GL, curr_membership);
            allocated_graphs.push_back(next_GL);
            current_GL = next_GL;

            for (int i = 0; i < vcount; i++) {
                membership[i] = curr_membership[membership[i]];
            }
        }
    }

    unordered_map<int, vector<node_t>> cpp_groups;
    for (int i = 0; i < vcount; ++i) {
        int comm_id = membership[i];
        node_t node_id = i + 1;
        cpp_groups[comm_id].push_back(node_id);
    }

    py::dict id_to_node = G_.id_to_node;

    py::list final_result;
    for (auto& kv : cpp_groups) {
        py::set py_comm;
        for (node_t node_id : kv.second) {
            py::object node_obj = id_to_node[py::cast(node_id)];
            py_comm.add(node_obj);
        }
        final_result.append(py_comm);
    }

    for (auto g : allocated_graphs) {
        delete g;
    }

    return final_result;
}