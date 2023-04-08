#include "centrality.h"
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"


void betweenness_dijkstra(const Graph_L& G_l, const int &S, std::vector<double>& bc, double cutoff) {
    int N = G_l.n;
    int edge_number_path = 0;
    __gnu_pbds::priority_queue<compare_node> q;
    std::vector<double> dis(N+1, INFINITY);
    std::vector<bool> vis(N+1, false);
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
    q.push(compare_node(S, 0));
    int cnt_St = 0;
    while(!q.empty()) {
        int u = q.top().x;
        q.pop();
        if (vis[u]){
            continue;
        }
        if (cutoff >= 0 && dis[u] > cutoff){
            continue;
        }
        St[cnt_St++] = u;
        vis[u] = true;
        for(int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            if(cutoff >= 0 && (dis[u] + E[p].w) > cutoff){
                continue;
            }
            if (dis[v] > dis[u] + E[p].w) {
                dis[v] = dis[u] + E[p].w;
                q.push(compare_node(v, dis[v]));
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
    while (cnt_St > 0) {
        int u = St[--cnt_St];
        float coeff = (1.0 + delta[u]) / count_path[u];
        for(int p = head_path[u]; p; p = E_path[p].next){
            delta[E_path[p].to] += count_path[E_path[p].to] * coeff;
        }

        if (u != S)
            bc[u] += delta[u];
    }
}

py::object betweenness_centrality(py::object G, py::object weight, py::object cutoff){
    Graph& G_ = G.cast<Graph&>();
    double cutoff_ = -1;
    if (!cutoff.is_none()){
        cutoff_ = cutoff.cast<double>();
    }
    int N = G_.node.size();
    bool is_directed = G.attr("is_directed")().cast<bool>();
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

    std::vector<double> bc(N+1, 0);
    py::list res_lst = py::list();
    for (int i = 1; i <= N; ++i){

        betweenness_dijkstra(G_l, i, bc, cutoff_);
    }
    double scale = 1.0;
    if(!is_directed){
        scale = 0.5;
    }
    for(int i = 1; i <= N; i++){
        res_lst.append(scale * bc[i]);
    }
    return res_lst;
}