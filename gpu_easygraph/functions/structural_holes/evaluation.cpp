#include <string>
#include <vector>
#include <memory> // 需要包含memory头文件

#include "structural_holes/evaluation.cuh"
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


int constraint(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<double>& W,
    _OUT_ std::vector<double>& constraint
) {
    // printf("V size: %zu, E size: %zu, W size: %zu, v_id: %d\n", V.size(), E.size(), W.size(), v_id);
    int len_V = V.size() - 1;
    int len_E = E.size();
    int warp_size = decide_warp_size(len_V, len_E);
    
    constraint = vector<double>(len_V);
    int r = cuda_constraint(V.data(), E.data(), W.data(), len_V, len_E, constraint.data());

    return r;  // 成功
}

} // namespace gpu_easygraph