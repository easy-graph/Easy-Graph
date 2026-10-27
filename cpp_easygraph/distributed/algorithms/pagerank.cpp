#include "pagerank.h"

namespace egdist {
PageRankResult pagerank(const DistributedGraph& graph, double alpha, int max_iter,
                        double tol, bool weighted) {
    return PageRankWorker(graph).run(alpha, max_iter, tol, weighted);
}
}
