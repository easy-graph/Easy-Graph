#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <pybind11/pybind11.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "pagerank.h"
#include "../../classes/directed_graph.h"
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"

namespace py = pybind11;

py::object _pagerank(py::object G, double alpha, int max_iterator, double threshold, py::object weight) {

    bool is_directed = G.attr("is_directed")().cast<bool>();
    std::string weight_key = weight_to_string(weight);
    bool has_weight_key = !weight.is_none() && !weight_key.empty();

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

    const std::vector<LinkEdge>& E = G_l_ptr->edges;
    const std::vector<int>& outDegree = G_l_ptr->degree;
    const std::vector<int>& head = G_l_ptr->head;

    bool actually_weighted = false;
    std::vector<double> outWeightSum(N + 1, 0.0);

    if (has_weight_key) {
        #pragma omp parallel for reduction(|:actually_weighted)
        for (int i = 1; i <= N; ++i) {
            double sum_w = 0.0;
            for (int p = head[i]; p != -1; p = E[p].next) {
                sum_w += E[p].w;
                if (!actually_weighted && std::abs(E[p].w - 1.0) > 1e-9) {
                    actually_weighted = true; 
                }
            }
            outWeightSum[i] = sum_w;
        }
    }
    bool use_weighted_logic = has_weight_key && actually_weighted;

    std::vector<double> oldPR(N + 1, 1.0 / N);
    std::vector<double> newPR(N + 1, 0.0);
    int cnt = 0;

    while (cnt < max_iterator) {
        double dangling_sum = 0.0;

        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 1; i <= N; ++i) {
            bool is_dangling = use_weighted_logic ? (outWeightSum[i] < 1e-15) : (outDegree[i] == 0);
            if (is_dangling) dangling_sum += oldPR[i];
        }

        if (!use_weighted_logic) {
            #pragma omp parallel for schedule(dynamic, 128)
            for (int i = 1; i <= N; ++i) {
                if (outDegree[i] == 0) continue;
                double out_val = (oldPR[i] / outDegree[i]) * alpha;
                for (int p = head[i]; p != -1; p = E[p].next) {
                    #pragma omp atomic
                    newPR[E[p].to] += out_val;
                }
            }
        } else {
            #pragma omp parallel for schedule(dynamic, 128)
            for (int i = 1; i <= N; ++i) {
                if (outWeightSum[i] < 1e-15) continue;
                double out_val = (oldPR[i] / outWeightSum[i]) * alpha;
                for (int p = head[i]; p != -1; p = E[p].next) {
                    #pragma omp atomic
                    newPR[E[p].to] += out_val * E[p].w;
                }
            }
        }

        double diff_sum = 0.0;
        double jump_val = (1.0 - alpha) / N + (dangling_sum / N) * alpha;

        #pragma omp parallel for reduction(+:diff_sum)
        for (int i = 1; i <= N; ++i) {
            double final_pr = newPR[i] + jump_val;
            diff_sum += std::fabs(final_pr - oldPR[i]);
            oldPR[i] = final_pr;
            newPR[i] = 0.0;
        }

        if (diff_sum < threshold * N) break;
        cnt++;
    }

    py::list res_lst;
    for (int i = 1; i <= N; ++i) res_lst.append(oldPR[i]);
    return res_lst;
}