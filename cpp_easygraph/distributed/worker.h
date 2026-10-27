#pragma once
#include <vector>
#include "distributed_graph.h"

namespace egdist {
struct PageRankResult {
    std::vector<VertexId> ids;
    std::vector<double> values;
    int iterations;
};


class PageRankWorker {
public:
    explicit PageRankWorker(const DistributedGraph& graph) : graph_(graph) {}
    PageRankResult run(double alpha, int max_iter, double tol, bool weighted) const;
private:
    const DistributedGraph& graph_;
};
}
