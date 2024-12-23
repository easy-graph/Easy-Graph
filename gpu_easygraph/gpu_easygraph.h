#include <vector>

#include "./common/err.h"

namespace gpu_easygraph {

int closeness_centrality(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    std::vector<double>& CC
);



int betweenness_centrality(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    bool is_directed,
    bool normalized,
    bool endpoints,
    std::vector<double>& BC
);



int k_core(
    const std::vector<int>& V,
    const std::vector<int>& E,
    std::vector<int>& KC
);



int sssp_dijkstra(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    int target,
    std::vector<double>& res
);



int pagerank(
    const std::vector<int>& V,
    const std::vector<int>& E,
    double alpha,
    int max_iter_num,
    double threshold,
    std::vector<double>& PR
);



int constraint(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask,
    std::vector<double>& constraint
);



int hierarchy(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& hierarchy
);



int effective_size(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& effective_size
);

} // namespace gpu_easygraph