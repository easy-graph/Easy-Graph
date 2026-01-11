#include <omp.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>
#include "pagerank.h"
#include "../../classes/directed_graph.h"
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"

struct IncomingEdge {
    int source;
    double weight;
};

struct Page {
    Page() {}
    Page(const double &_newPR, const double &_oldPR) { newPR = _newPR; oldPR = _oldPR; }
    double newPR, oldPR;
};

py::object _pagerank(py::object G, double alpha=0.85, int max_iterator=500, double threshold=1e-6, py::object weight=py::none()) {

    bool is_directed = G.attr("is_directed")().cast<bool>();
    bool use_weights = !weight.is_none();
    std::string weight_key = "";
    if (use_weights) {
        weight_key = weight_to_string(weight);
    }

    Graph_L* G_l_ptr = nullptr;
    int N = 0;

    if (is_directed) {
        DiGraph& G_ = G.cast<DiGraph&>();
        N = G_.node.size();
        if (G_.linkgraph_dirty) {
            G_.linkgraph_structure = graph_to_linkgraph(G_, true, weight_key, true, false);
            G_.linkgraph_dirty = false;
        }
        G_l_ptr = &G_.linkgraph_structure;
    } else {
        Graph& G_ = G.cast<Graph&>();
        N = G_.node.size();
        if (G_.linkgraph_dirty) {
            G_.linkgraph_structure = graph_to_linkgraph(G_, false, weight_key, true, false);
            G_.linkgraph_dirty = false;
        }
        G_l_ptr = &G_.linkgraph_structure;
    }

    std::vector<LinkEdge>& E = G_l_ptr->edges;
    std::vector<int>& outDegree = G_l_ptr->degree;
    std::vector<int>& head = G_l_ptr->head;

    std::vector<double> outWeightSum;
    if (use_weights) {
        outWeightSum.resize(N + 1, 0.0);
        #pragma omp parallel for
        for (int i = 1; i < N + 1; ++i) {
            if (outDegree[i] > 0) {
                double sum_w = 0.0;
                for (int p = head[i]; p != -1; p = E[p].next) {
                    sum_w += E[p].w;
                }
                outWeightSum[i] = sum_w;
            }
        }
    }

    std::vector<std::vector<IncomingEdge>> reverse_graph(N + 1);
    for (int u = 1; u < N + 1; ++u) {
        for (int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            double w = use_weights ? E[p].w : 1.0;
            reverse_graph[v].push_back({u, w});
        }
    }

    std::vector<Page> page(N + 1);
    #pragma omp parallel for
    for (int i = 1; i < N + 1; ++i) {
        page[i] = Page(0.0, 1.0 / N);
    }

    int cnt = 0;
    int shouldStop = 0;

    while (!shouldStop) {
        shouldStop = 1;
        double dangling_sum = 0.0;

        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 1; i < N + 1; ++i) {
            bool is_dangling = false;
            if (use_weights) {
                if (outDegree[i] == 0 || outWeightSum[i] == 0.0) is_dangling = true;
            } else {
                if (outDegree[i] == 0) is_dangling = true;
            }
            if (is_dangling) dangling_sum += page[i].oldPR;
        }

        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 1; i < N + 1; ++i) {
            double incoming_pr = 0.0;
            
            for (const auto& edge : reverse_graph[i]) {
                int source = edge.source;
                
                if (use_weights) {
                    if (outWeightSum[source] > 0) {
                        incoming_pr += page[source].oldPR * (edge.weight / outWeightSum[source]);
                    }
                } else {
                    if (outDegree[source] > 0) {
                        incoming_pr += page[source].oldPR / outDegree[source];
                    }
                }
            }

            page[i].newPR = (1.0 - alpha) / N + alpha * (dangling_sum / N + incoming_pr);
        }

        double diff_sum = 0.0;
        #pragma omp parallel for reduction(+:diff_sum)
        for (int i = 1; i < N + 1; ++i) {
            diff_sum += std::fabs(page[i].newPR - page[i].oldPR);
            page[i].oldPR = page[i].newPR;
            page[i].newPR = 0.0;
        }

        if (diff_sum > threshold * N) shouldStop = 0;
        cnt++;
        if (cnt >= max_iterator) break;
    }

    py::list res;
    for (int i = 1; i < N + 1; ++i) {
        res.append(page[i].oldPR);
    }

    return res;
}