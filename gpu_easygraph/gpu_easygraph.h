#include <vector>

#include "./common/err.h"

namespace gpu_easygraph {

int closeness_centrality(
    std::vector<int> V,
    std::vector<int> E,
    std::vector<double> W,
    std::vector<int> sources,
    std::vector<double>& CC
);



int betweenness_centrality(
    std::vector<int> V,
    std::vector<int> E,
    std::vector<double> W,
    std::vector<int> sources,
    bool is_directed,
    bool normalized,
    bool endpoints,
    std::vector<double>& BC
);



int k_core(
    std::vector<int> V,
    std::vector<int> E,
    std::vector<int>& KC
);

} // namespace gpu_easygraph