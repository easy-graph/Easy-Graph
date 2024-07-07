#include <limits>
#include <vector>

#include "path/sssp_dijkstra.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

static int decide_warp_size(
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



int sssp_dijkstra(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<double>& W,
    _IN_ const vector<int>& sources,
    _IN_ int target,
    _OUT_ vector<double>& res
)
{
    int len_V = V.size() - 1;
    int len_E = E.size();

    int warp_size = decide_warp_size(len_V, len_E);

    res = vector<double>(sources.size() * V.size());

    int r = cuda_sssp_dijkstra(V.data(), E.data(), W.data(),
            sources.data(), len_V, len_E, sources.size(),
            target, warp_size, res.data());

    double double_inf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < res.size(); ++i) {
        if (res[i] >= EG_DOUBLE_INF) {
            res[i] = double_inf;
        }
    }

    return r;
}

} // namespace gpu_easygraph