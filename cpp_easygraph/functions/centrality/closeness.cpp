#include "centrality.h"
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"

double closeness_dijkstra(const Graph_L& G_l, const int &S){
    int N = G_l.n;
    __gnu_pbds::priority_queue<compare_node> q;
    std::vector<int> dis(N+1, INFINITY);
    std::vector<bool> vis(N+1, false);
    const std::vector<LinkEdge>& E = G_l.edges;
    const std::vector<int>& head = G_l.head;
    int number_connected = 0;
    long long sum_dis = 0;
    dis[S] = 0; 
    q.push(compare_node(S, 0));
    while(!q.empty()) {
        int u=q.top().x;
        q.pop();
        if (vis[u]){
            continue;
        } 
        vis[u] = true;
        number_connected += 1;
        sum_dis += dis[u];
        for(register int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            if (dis[v] > dis[u] + E[p].w) {
                dis[v] = dis[u] + E[p].w;
                q.push(compare_node(v, dis[v]));
            }
        }
    }

    float res;
    if (number_connected == 1)
        res = 0.0;
    else
        res = 1.0 * (number_connected - 1) * (number_connected - 1) / ((N - 1) * sum_dis);
    
    return res;
}

py::object closeness_centrality(py::object G, py::object weight) {
    Graph& G_ = G.cast<Graph&>();
    int N = G_.node.size();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    std::string weight_key = weight_to_string(weight);
    // Graph_L G_l = graph_to_linkgraph(G_, is_directed, weight_key, false, true);
    const Graph_L& G_l = graph_to_linkgraph(G_, is_directed, weight_key, false, true);

    py::list res_lst = py::list();
    for(register int i = 1; i <= N; i++){
        res_lst.append(closeness_dijkstra(G_l, i));
    }
    return res_lst;
}