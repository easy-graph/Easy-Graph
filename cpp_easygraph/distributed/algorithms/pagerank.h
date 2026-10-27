#pragma once
#include "../worker.h"

namespace egdist {
PageRankResult pagerank(const DistributedGraph& graph, double alpha, int max_iter,
                        double tol, bool weighted);
}
