#include <vector>
#include <string>

#include "centrality/closeness_centrality.cuh"
#include "centrality/betweenness_centrality.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::pair;
using std::string;
using std::vector;

static int decide_warp_size (
    _IN_ int len_V,
    _IN_ int len_E
)
{
    vector<int> warp_size_cand{1, 2, 4, 8, 16, 32};

    if (len_E / len_V < warp_size_cand.front()) {
        return warp_size_cand.front();
    }

    for (int i = 0; i + 1 < warp_size_cand.size(); ++i) {
        if (warp_size_cand[i] <= len_E / len_V
                && len_E / len_V < warp_size_cand[i + 1]) {
            return warp_size_cand[i + 1];
        }
    }
    return warp_size_cand.back();
}



int closeness_centrality(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _IN_ const std::vector<double>& W,
    _IN_ const std::vector<int>& sources,
    _OUT_ std::vector<double>& CC
) {
    int len_V = V.size() - 1;
    int len_E = E.size();

    int warp_size = decide_warp_size(len_V, len_E);
    
    CC = vector<double>(len_V);

    int r = cuda_closeness_centrality(V.data(), E.data(), W.data(), 
            sources.data(), len_V, len_E, sources.size(), 
            warp_size, CC.data());
        
    return r;
}



int betweenness_centrality(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _IN_ const std::vector<double>& W,
    _IN_ const std::vector<int>& sources,
    _IN_ bool is_directed,
    _IN_ bool normalized,
    _IN_ bool endpoints,
    _OUT_ std::vector<double>& BC
) {
    int len_V = V.size() - 1;
    int len_E = E.size();

    int warp_size = decide_warp_size(len_V, len_E);

    BC = vector<double>(len_V);

    int r = cuda_betweenness_centrality(V.data(), E.data(), W.data(),
            sources.data(), len_V, len_E, sources.size(),
            warp_size, is_directed, normalized, endpoints, BC.data());

    return r;
}

} // namespace gpu_easygraph