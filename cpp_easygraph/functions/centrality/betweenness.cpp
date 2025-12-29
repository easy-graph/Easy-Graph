#include <omp.h>
#include <vector>
#include <queue>
#include <limits.h>
#include <algorithm>
#include <string>
#include <cstdio>

#include "centrality.h"
#ifdef EASYGRAPH_ENABLE_GPU
#include <gpu_easygraph.h>
#endif

#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"

namespace py = pybind11;

void betweenness_bfs_worker(
    const Graph_L& G_l, const int& S, std::vector<double>& bc, int cutoff, int endpoints_,
    std::vector<int>& q, std::vector<int>& dis, std::vector<int>& head_path, std::vector<int>& St,
    std::vector<long long>& count_path, std::vector<double>& delta, std::vector<LinkEdge>& E_path,
    std::vector<int>& stamp, int& cur_stamp
) {
    int N = G_l.n;
    int edge_number_path = 0;
    int cnt_St = 0;
    ++cur_stamp;
    if ((int)q.size() < N + 1)
        q.resize(N + 1);
    int front = 0, back = 0;
    int cutoff_int = (cutoff < 0) ? -1 : cutoff;
    
    stamp[S] = cur_stamp;
    dis[S] = 0;
    count_path[S] = 1;
    delta[S] = 0.0;
    head_path[S] = 0;
    q[back++] = S;

    const std::vector<int>& head = G_l.head;
    const std::vector<LinkEdge>& E = G_l.edges;

    while (front < back) {
        int u = q[front++];
        int du = dis[u];
        if (cutoff_int >= 0 && du > cutoff_int)
            break;
        St[cnt_St++] = u;

        for (int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            int new_dis = du + 1;
            if (cutoff_int >= 0 && new_dis > cutoff_int)
                continue;

            if (stamp[v] != cur_stamp) {
                stamp[v] = cur_stamp;
                dis[v] = new_dis;
                count_path[v] = count_path[u];
                delta[v] = 0.0;
                head_path[v] = 0;
                q[back++] = v;
                E_path[++edge_number_path].next = head_path[v];
                E_path[edge_number_path].to = u;
                head_path[v] = edge_number_path;
            } else if (dis[v] == new_dis) {
                count_path[v] += count_path[u];
                E_path[++edge_number_path].next = head_path[v];
                E_path[edge_number_path].to = u;
                head_path[v] = edge_number_path;
            }
        }
    }

    if (endpoints_)
        bc[S] += cnt_St - 1;

    while (cnt_St > 0) {
        int u = St[--cnt_St];
        double cu = count_path[u];
        if (cu != 0) {
            double coeff = (1.0 + delta[u]) / cu;
            for (int p = head_path[u]; p; p = E_path[p].next) {
                int w = E_path[p].to;
                delta[w] += count_path[w] * coeff;
            }
        }
        if (u != S)
            bc[u] += delta[u] + endpoints_;
    }
}

void betweenness_dijkstra_worker(
    const Graph_L& G_l, const int& S, std::vector<double>& bc, double cutoff,
    std::vector<int>& dis, std::vector<int>& head_path,
    std::vector<int>& St, std::vector<long long>& count_path, std::vector<double>& delta,
    std::vector<LinkEdge>& E_path, int endpoints_,
    std::vector<int>& stamp, int& cur_stamp
) {
    const int dis_inf = 0x3f3f3f3f;
    
    int N = G_l.n;
    int edge_number_path = 0;
    int cnt_St = 0;
    ++cur_stamp;

    stamp[S] = cur_stamp;
    dis[S] = 0;
    count_path[S] = 1;
    delta[S] = 0.0;
    head_path[S] = 0;

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    pq.push({0, S});

    const std::vector<int>& head = G_l.head;
    const std::vector<LinkEdge>& E = G_l.edges;

    while (!pq.empty()) {
        std::pair<int, int> top = pq.top();
        pq.pop();
        int d = top.first;
        int u = top.second;

        if (d > dis[u]) continue;
        
        if (cutoff >= 0 && d > cutoff) continue;
        
        St[cnt_St++] = u;

        for (int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            int w = E[p].w;
            int nd = dis[u] + w;
            
            if (cutoff >= 0 && nd > cutoff) continue;

            bool first_visit = (stamp[v] != cur_stamp);
            
            if (first_visit || dis[v] > nd) {
                if (first_visit) {
                    stamp[v] = cur_stamp;
                    delta[v] = 0.0;
                }
                dis[v] = nd;
                count_path[v] = count_path[u];
                head_path[v] = 0;
                E_path[++edge_number_path].next = head_path[v];
                E_path[edge_number_path].to = u;
                head_path[v] = edge_number_path;
                
                pq.push({nd, v});
            } else if (dis[v] == nd) {
                count_path[v] += count_path[u];
                E_path[++edge_number_path].next = head_path[v];
                E_path[edge_number_path].to = u;
                head_path[v] = edge_number_path;
            }
        }
    }

    if (endpoints_)
        bc[S] += cnt_St - 1;

    while (cnt_St > 0) {
        int u = St[--cnt_St];
        double cu = count_path[u];
        if (cu != 0) {
            double coeff = (1.0 + delta[u]) / cu;
            for (int p = head_path[u]; p; p = E_path[p].next) {
                int w = E_path[p].to;
                delta[w] += count_path[w] * coeff;
            }
        }
        if (u != S)
            bc[u] += delta[u] + endpoints_;
    }
}

static double calc_scale(int len_V, int is_directed, int normalized, int endpoints) {
    double scale = 1.0;
    if (normalized) {
        if (endpoints) {
            if (len_V < 2) {
                scale = 1.0;
            } else {
                scale = 1.0 / (double(len_V) * (len_V - 1));
            }
        } else {
            if (len_V <= 2) {
                scale = 1.0;
            } else {
                scale = 1.0 / ((double(len_V) - 1) * (len_V - 2));
            }
        }
    } else {
        if (!is_directed) {
            scale = 0.5;
        } else {
            scale = 1.0;
        }
    }
    return scale;
}

static py::object invoke_cpp_betweenness_centrality(
    py::object G, py::object weight, py::object cutoff, py::object sources,
    py::object normalized, py::object endpoints
) {
    Graph& G_ = G.cast<Graph&>();
    int cutoff_ = -1;
    if (!cutoff.is_none()) {
        cutoff_ = cutoff.cast<int>();
    }
    int N = G_.node.size();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    int normalized_ = normalized.cast<bool>();
    int endpoints_ = endpoints.cast<bool>();
    double scale = calc_scale(N, is_directed, normalized_, endpoints_);
    bool use_weights = !weight.is_none();
    std::string weight_key = "";
    if (use_weights) {
        weight_key = weight_to_string(weight);
    }

    Graph_L G_l;
    if (G_.linkgraph_dirty) {
        G_l = graph_to_linkgraph(G_, is_directed, weight_key, false, false);
        G_.linkgraph_structure = G_l;
    } else {
        G_l = G_.linkgraph_structure;
    }

    int edges_num = G_l.edges.size();
    std::vector<double> bc(N + 1, 0.0);
    std::vector<double> BC;
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    std::vector<std::vector<int>> dis_all(num_threads, std::vector<int>(N + 1));
    std::vector<std::vector<int>> head_path_all(num_threads, std::vector<int>(N + 1));
    std::vector<std::vector<int>> St_all(num_threads, std::vector<int>(N + 1));
    std::vector<std::vector<long long>> count_path_all(num_threads, std::vector<long long>(N + 1));
    std::vector<std::vector<double>> delta_all(num_threads, std::vector<double>(N + 1));
    std::vector<std::vector<LinkEdge>> E_path_all(num_threads, std::vector<LinkEdge>(edges_num + 1));
    
    std::vector<std::vector<int>> queue_all(num_threads, std::vector<int>(N + 1));
    std::vector<std::vector<int>> stamp_all(num_threads, std::vector<int>(N + 1, 0));
    std::vector<int> cur_stamp_all(num_threads, 0);

    std::vector<std::vector<double>> bc_local_all(num_threads, std::vector<double>(N + 1, 0.0));

    if (!sources.is_none()) {
        py::list sources_list = py::list(sources);
        int sources_list_len = py::len(sources_list);
        std::vector<node_t> sources_vec;
        sources_vec.reserve(sources_list_len);
        for (int i = 0; i < sources_list_len; i++) {
            if (G_.node_to_id.attr("get")(sources_list[i], py::none()).is_none()) {
                printf("The node should exist in the graph!");
                return py::none();
            }
            sources_vec.push_back(G_.node_to_id.attr("get")(sources_list[i]).cast<node_t>());
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < sources_list_len; i++) {
            node_t source_id = sources_vec[i];
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            auto& bc_local = bc_local_all[tid];
            auto& dis = dis_all[tid];
            auto& head_path = head_path_all[tid];
            auto& St = St_all[tid];
            auto& count_path = count_path_all[tid];
            auto& delta = delta_all[tid];
            auto& E_path = E_path_all[tid];
            auto& q = queue_all[tid];
            auto& stamp = stamp_all[tid];
            int& cur_stamp = cur_stamp_all[tid];

            if (use_weights) {
                betweenness_dijkstra_worker(
                    G_l, source_id, bc_local, cutoff_, dis, head_path,
                    St, count_path, delta, E_path, endpoints_, stamp, cur_stamp
                );
            } else {
                betweenness_bfs_worker(
                    G_l, source_id, bc_local, cutoff_, endpoints_, q, dis, head_path,
                    St, count_path, delta, E_path, stamp, cur_stamp
                );
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 1; i <= N; ++i) {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            auto& bc_local = bc_local_all[tid];
            auto& dis = dis_all[tid];
            auto& head_path = head_path_all[tid];
            auto& St = St_all[tid];
            auto& count_path = count_path_all[tid];
            auto& delta = delta_all[tid];
            auto& E_path = E_path_all[tid];
            auto& q = queue_all[tid];
            auto& stamp = stamp_all[tid];
            int& cur_stamp = cur_stamp_all[tid];

            if (use_weights) {
                betweenness_dijkstra_worker(
                    G_l, i, bc_local, cutoff_, dis, head_path,
                    St, count_path, delta, E_path, endpoints_, stamp, cur_stamp
                );
            } else {
                betweenness_bfs_worker(
                    G_l, i, bc_local, cutoff_, endpoints_, q, dis, head_path,
                    St, count_path, delta, E_path, stamp, cur_stamp
                );
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (int j = 1; j <= N; ++j) {
        double s = 0.0;
        for (int tid = 0; tid < num_threads; ++tid)
            s += bc_local_all[tid][j];
        bc[j] += s;
    }
#else
    for (int j = 1; j <= N; ++j) {
        bc[j] += bc_local_all[0][j];
    }
#endif

    BC.reserve(N);
    for (int i = 1; i <= N; i++) {
        BC.push_back(scale * bc[i]);
    }

    py::array::ShapeContainer ret_shape{(int)BC.size()};
    py::array_t<double> ret(ret_shape, BC.data());
    return ret;
}

#ifdef EASYGRAPH_ENABLE_GPU
static py::object invoke_gpu_betweenness_centrality(py::object G, py::object weight,
py::object py_sources, py::object normalized, py::object endpoints) {
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
    auto sources = G_.gen_CSR_sources(py_sources);
    std::vector<double> BC;
    bool is_directed = G.attr("is_directed")().cast<bool>();
    int gpu_r = gpu_easygraph::betweenness_centrality(V, E, *W_p, *sources,
        is_directed, normalized.cast<py::bool_>(),
        endpoints.cast<py::bool_>(), BC);
    if (gpu_r != gpu_easygraph::EG_GPU_SUCC) {
        // the code below will throw an exception
        py::pybind11_fail(gpu_easygraph::err_code_detail(gpu_r));
    }
    py::array::ShapeContainer ret_shape{(int)BC.size()};
    py::array_t<double> ret(ret_shape, BC.data());
    return ret;
}
#endif


py::object betweenness_centrality(py::object G, py::object weight, py::object cutoff, py::object sources,
py::object normalized, py::object endpoints) {
#ifdef EASYGRAPH_ENABLE_GPU
    return invoke_gpu_betweenness_centrality(G, weight, sources, normalized, endpoints);
#else
    return invoke_cpp_betweenness_centrality(G, weight, cutoff, sources, normalized, endpoints);
#endif
}

// void betweenness_dijkstra(const Graph_L& G_l, const int &S, std::vector<double>& bc, double cutoff) {
//     int N = G_l.n;
//     int edge_number_path = 0;
//     __gnu_pbds::priority_queue<compare_node> q;
//     std::vector<double> dis(N+1, INFINITY);
//     std::vector<bool> vis(N+1, false);
//     std::vector<int> head_path(N+1, 0);
//     const std::vector<int>& head = G_l.head;
//     const std::vector<LinkEdge>& E = G_l.edges;
//     int edges_num = E.size();
//     std::vector<int> St(N+1, 0);
//     std::vector<long long> count_path(N+1, 0);
//     std::vector<double> delta(N+1, 0);
//     std::vector<LinkEdge> E_path(edges_num+1);
//     head_path[S] = 0;
//     dis[S] = 0;
//     count_path[S] = 1;
//     dis[S] = 0;
//     count_path[S] = 1;
//     q.push(compare_node(S, 0));
//     int cnt_St = 0;
//     while(!q.empty()) {
//         int u = q.top().x;
//         q.pop();
//         if (vis[u]){
//             continue;
//         }
//         if (cutoff >= 0 && dis[u] > cutoff){
//             continue;
//         }
//         St[cnt_St++] = u;
//         vis[u] = true;
//         for(int p = head[u]; p != -1; p = E[p].next) {
//             int v = E[p].to;
//             if(cutoff >= 0 && (dis[u] + E[p].w) > cutoff){
//                 continue;
//             }
//             if (dis[v] > dis[u] + E[p].w) {
//                 dis[v] = dis[u] + E[p].w;
//                 q.push(compare_node(v, dis[v]));
//                 count_path[v] = count_path[u];
//                 head_path[v] = 0;
//                 E_path[++edge_number_path].next = head_path[v];
//                 E_path[edge_number_path].to = u;
//                 head_path[v] = edge_number_path;
//             }
//             else if (dis[v] == dis[u] + E[p].w) {
//                 count_path[v] += count_path[u];
//                 E_path[++edge_number_path].next = head_path[v];
//                 E_path[edge_number_path].to = u;
//                 head_path[v] = edge_number_path;
//             }
//         }
//     }
//     while (cnt_St > 0) {
//         int u = St[--cnt_St];
//         float coeff = (1.0 + delta[u]) / count_path[u];
//         for(int p = head_path[u]; p; p = E_path[p].next){
//             delta[E_path[p].to] += count_path[E_path[p].to] * coeff;
//         }
//         if (u != S)
//             bc[u] += delta[u];
//     }
// }
