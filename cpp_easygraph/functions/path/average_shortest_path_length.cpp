#include "path.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"

#include <vector>
#include <queue>
#include <limits>
#include <string>

double _bfs_sum(const Graph_L& G_l, int source) {
    int N = G_l.n;
    std::vector<int> dis(N + 1, -1);
    std::queue<int> q;

    dis[source] = 0;
    q.push(source);

    double sum = 0.0;
    int visited_count = 0;

    const std::vector<int>& head = G_l.head;
    const std::vector<LinkEdge>& E = G_l.edges;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        sum += dis[u];
        visited_count++;

        for (int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            if (dis[v] == -1) {
                dis[v] = dis[u] + 1;
                q.push(v);
            }
        }
    }
    return (visited_count == N) ? sum : -1.0;
}

double _dijkstra_sum(const Graph_L& G_l, int source) {
    int N = G_l.n;
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dis(N + 1, INF);
    std::priority_queue<std::pair<double, int>, 
                        std::vector<std::pair<double, int>>, 
                        std::greater<std::pair<double, int>>> pq;

    dis[source] = 0.0;
    pq.push({0.0, source});

    double sum = 0.0;
    int visited_count = 0;

    const std::vector<int>& head = G_l.head;
    const std::vector<LinkEdge>& E = G_l.edges;

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dis[u]) continue;
        sum += d;
        visited_count++;

        for (int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            double w = static_cast<double>(E[p].w); 
            if (dis[u] + w < dis[v]) {
                dis[v] = dis[u] + w;
                pq.push({dis[v], v});
            }
        }
    }
    return (visited_count == N) ? sum : -1.0;
}


py::object average_shortest_path_length(py::object G, py::object weight, py::object method) {
    Graph& G_ = G.cast<Graph&>();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    
    std::string weight_key = "";
    if (!weight.is_none()) {
        weight_key = weight.cast<std::string>();
    }

    std::string method_str;
    if (method.is_none()) {
        method_str = weight.is_none() ? "single_source_bfs" : "dijkstra";
    } else {
        method_str = method.cast<std::string>();
    }

    if(G_.linkgraph_dirty){
        G_.linkgraph_structure = graph_to_linkgraph(G_, is_directed, weight_key, true, false);
        G_.linkgraph_dirty = false;
    }
    const Graph_L& G_l = G_.linkgraph_structure;

    int N = G_l.n;
    if (N <= 1) return py::float_(0.0);

    double total_sum = 0.0;
    bool is_connected = true;

    {
        py::gil_scoped_release release;

        #pragma omp parallel for reduction(+:total_sum) schedule(dynamic)
        for (int i = 1; i <= N; i++) {
            if (!is_connected) continue; 

            double local_sum = 0.0;
            if (method_str == "single_source_bfs" || method_str == "unweighted") {
                local_sum = _bfs_sum(G_l, i);
            } else {
                local_sum = _dijkstra_sum(G_l, i);
            }

            if (local_sum < 0) {
                #pragma omp critical
                is_connected = false;
            } else {
                total_sum += local_sum;
            }
        }
    }

    if (!is_connected) {  
        if (is_directed) {
            throw py::value_error("Graph is not strongly connected.");
        } 
        else {
            throw py::value_error("Graph is not connected.");
        }
    }

    return py::float_(total_sum / (static_cast<double>(N) * (N - 1)));
}