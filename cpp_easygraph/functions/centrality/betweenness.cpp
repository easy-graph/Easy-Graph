#include "centrality.h"

#ifdef EASYGRAPH_ENABLE_GPU
#include <gpu_easygraph.h>
#endif

#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include "../../classes/segment_tree.cpp"


void betweenness_dijkstra(const Graph_L& G_l, const int &S, std::vector<double>& bc, double cutoff, Segment_tree_zkw& segment_tree_zkw, int endpoints_) {
    const int dis_inf = 0x3f3f3f3f;
    int N = G_l.n;
    int edge_number_path = 0;
    segment_tree_zkw.init(N);
    std::vector<int> dis(N+1, INT_MAX);
    std::vector<int> head_path(N+1, 0);
    const std::vector<int>& head = G_l.head;
    const std::vector<LinkEdge>& E = G_l.edges;
    int edges_num = E.size();
    std::vector<int> St(N+1, 0);
    std::vector<long long> count_path(N+1, 0);
    std::vector<double> delta(N+1, 0);
    std::vector<LinkEdge> E_path(edges_num+1);
    head_path[S] = 0;
    dis[S] = 0; 
    count_path[S] = 1; 
    segment_tree_zkw.change(S, 0);
    int cnt_St = 0;
   
    while(segment_tree_zkw.t[1] != dis_inf) {
        int u = segment_tree_zkw.num[1];
        if(u==0) break;
        segment_tree_zkw.change(u, dis_inf);
        if (cutoff >= 0 && dis[u] > cutoff){
            continue;
        }
        St[cnt_St++] = u;        
        for(int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            if(cutoff >= 0 && (dis[u] + E[p].w) > cutoff){
                continue;
            }
            if (dis[v] > dis[u] + E[p].w) {
                dis[v] = dis[u] + E[p].w;  
                segment_tree_zkw.change(v, dis[v]);    
                count_path[v] = count_path[u];
                head_path[v] = 0;
                E_path[++edge_number_path].next = head_path[v];
                E_path[edge_number_path].to = u;
                head_path[v] = edge_number_path;
            }
            else if (dis[v] == dis[u] + E[p].w) {
                count_path[v] += count_path[u];
                E_path[++edge_number_path].next = head_path[v];
                E_path[edge_number_path].to = u;
                head_path[v] = edge_number_path;
                
            }
        }
    }
    if (endpoints_) {
        bc[S] += cnt_St - 1;
    }
    while (cnt_St > 0) {
        int u = St[--cnt_St];
        float coeff = (1.0 + delta[u]) / count_path[u];
        for(int p = head_path[u]; p; p = E_path[p].next){
            delta[E_path[p].to] += count_path[E_path[p].to] * coeff;
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
        } else if (len_V <= 2) {
            scale = 1.0;
        } else {
            scale = 1.0 / ((double(len_V) - 1) * (len_V - 2));
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



static py::object invoke_cpp_betweenness_centrality(py::object G, py::object weight, 
                                    py::object cutoff, py::object sources, 
                                    py::object normalized, py::object endpoints){
    Graph& G_ = G.cast<Graph&>();
    int cutoff_ = -1;
    if (!cutoff.is_none()){
        cutoff_ = cutoff.cast<int>();
    }
    int N = G_.node.size();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    int normalized_ = normalized.cast<bool>();
    int endpoints_ = endpoints.cast<bool>();
    double scale = calc_scale(N, is_directed, normalized_, endpoints_);
    std::string weight_key = weight_to_string(weight);
    Graph_L G_l;
    if(G_.linkgraph_dirty){
        G_l = graph_to_linkgraph(G_, is_directed, weight_key, false, false);
        G_.linkgraph_structure=G_l;
        G_.linkgraph_dirty = false;
    }
    else{
        G_l = G_.linkgraph_structure;
    }
    Segment_tree_zkw segment_tree_zkw(N);
    std::vector<double> bc(N+1, 0);
    std::vector<double> BC;
    if(!sources.is_none()){
        py::list sources_list = py::list(sources);
        int sources_list_len = py::len(sources_list);
        for(register int i = 0; i < sources_list_len; i++){
            if(G_.node_to_id.attr("get")(sources_list[i],py::none()).is_none()){
                printf("The node should exist in the graph!");
                return py::none();
            }
            node_t source_id = G_.node_to_id.attr("get")(sources_list[i]).cast<node_t>();
            betweenness_dijkstra(G_l, source_id, bc, cutoff_, segment_tree_zkw, endpoints_);
        }
        for(int i = 1; i <= N; i++){
            BC.push_back(scale * bc[i]);
        }
    }
    else{
        for (int i = 1; i <= N; ++i){
            betweenness_dijkstra(G_l, i, bc, cutoff_,segment_tree_zkw, endpoints_);
        }
        for(int i = 1; i <= N; i++){
            BC.push_back(scale * bc[i]);
        }
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

