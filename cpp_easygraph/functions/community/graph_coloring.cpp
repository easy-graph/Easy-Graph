#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <cstdint>
#include <chrono>
#include <queue>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#else
#warning "OpenMP is not available: omp_graph_coloring will fall back to single-threaded execution."
#endif

#include "../../classes/linkgraph.h"

using namespace std;

vector<int> greedy_graph_coloring(const Graph_L& G) {
    int n = G.n;
    if (n == 0) return vector<int>();

    vector<int> degree(n + 1, 0);
    for (int v = 1; v <= n; ++v) {
        for (int e = G.head[v]; e != -1; e = G.edges[e].next) {
            degree[v]++;
        }
    }

    vector<int> nodes(n);
    iota(nodes.begin(), nodes.end(), 1);
    sort(nodes.begin(), nodes.end(), [&degree](int a, int b) {
        return degree[a] > degree[b];
    });

    vector<int> colors(n, -1);
    vector<int> used(n + 1, -1);

    for (int u : nodes) {
        for (int e = G.head[u]; e != -1; e = G.edges[e].next) {
            int v = G.edges[e].to;
            if (v == u) continue;
            int neighbor_color = colors[v - 1];
            if (neighbor_color != -1) {
                used[neighbor_color] = u;
            }
        }

        int cr = 0;
        while (cr <= n && used[cr] == u) {
            cr++;
        }
        colors[u - 1] = cr;
    }

    return colors;
}

vector<int> omp_graph_coloring(const Graph_L& G) {
    int n = G.n;
    if (n == 0) return vector<int>();

    vector<int> degree(n + 1, 0);
    for (int v = 1; v <= n; ++v) {
        for (int e = G.head[v]; e != -1; e = G.edges[e].next) {
            degree[v]++;
        }
    }

    vector<int> nodes(n);
    iota(nodes.begin(), nodes.end(), 1);
    sort(nodes.begin(), nodes.end(), [&degree](int a, int b) {
        return degree[a] > degree[b];
    });

    vector<int> colors(n, -1);

    #pragma omp parallel
    {
        vector<int> local_used(n + 1, -1);
        
        #pragma omp for schedule(guided)
        for (int i = 0; i < n; ++i) {
            int u = nodes[i];
            
            fill(local_used.begin(), local_used.end(), -1);
            
            for (int e = G.head[u]; e != -1; e = G.edges[e].next) {
                int v = G.edges[e].to;
                if (v == u) continue;
                int neighbor_color = colors[v - 1];
                if (neighbor_color != -1 && neighbor_color <= n) {
                    local_used[neighbor_color] = u;
                }
            }
            
            int cr = 0;
            while (cr <= n && local_used[cr] == u) {
                cr++;
            }
            colors[u - 1] = cr;
        }
    }

    return colors;
}

bool verify_coloring(const Graph_L& g, const vector<int>& colors) {
    int n = g.n;
    for (int v = 1; v <= n; v++) {
        int v_color = colors[v - 1];
        for (int e = g.head[v]; e != -1; e = g.edges[e].next) {
            int u = g.edges[e].to;
            if (u == v) continue;
            int u_color = colors[u - 1];
            if (v_color == u_color) {
                return false;
            }
        }
    }
    return true;
}

int count_colors(const vector<int>& colors) {
    int max_color = -1;
    for (int c : colors) {
        if (c > max_color) {
            max_color = c;
        }
    }
    return max_color + 1;
}