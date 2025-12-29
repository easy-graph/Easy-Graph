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

    std::vector<Page> page(N + 1);
    #pragma omp parallel for
    for (int i = 1; i < N + 1; ++i) {
        page[i] = Page(0.0, 1.0 / N);
    }

    int cnt = 0;
    int shouldStop = 0;

    while (!shouldStop) {
        shouldStop = 1;
        double res = 0.0;

        #pragma omp parallel for reduction(+:res)
        for (int i = 1; i < N + 1; ++i) {
            bool is_dangling = false;
            if (use_weights) {
                if (outDegree[i] == 0 || outWeightSum[i] == 0.0) is_dangling = true;
            } else {
                if (outDegree[i] == 0) is_dangling = true;
            }
            if (is_dangling) res += page[i].oldPR;
        }

        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 1; i < N + 1; ++i) {
            if (use_weights) {
                if (outDegree[i] == 0 || outWeightSum[i] == 0.0) continue;
            } else {
                if (outDegree[i] == 0) continue;
            }

            if (!use_weights) {
                double tmpPR = (page[i].oldPR / outDegree[i]) * alpha;
                for (int p = head[i]; p != -1; p = E[p].next) {
                    #pragma omp atomic
                    page[E[p].to].newPR += tmpPR;
                }
            } else {
                double basePR = page[i].oldPR * alpha;
                double inv_sum = 1.0 / outWeightSum[i];
                for (int p = head[i]; p != -1; p = E[p].next) {
                    double contribution = basePR * (E[p].w * inv_sum);
                    #pragma omp atomic
                    page[E[p].to].newPR += contribution;
                }
            }
        }

        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (int i = 1; i < N + 1; ++i) {
            page[i].newPR += (1.0 - alpha) / N + (res / N) * alpha;
            sum += std::fabs(page[i].newPR - page[i].oldPR);
            page[i].oldPR = page[i].newPR;
            page[i].newPR = 0.0;
        }

        if (sum > threshold * N) shouldStop = 0;
        cnt++;
        if (cnt >= max_iterator) break;
    }

    py::list res_lst;
    for (int i = 1; i < N + 1; ++i) {
        res_lst.append(page[i].oldPR);
    }

    return res_lst;
}
