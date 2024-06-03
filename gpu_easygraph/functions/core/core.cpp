#include <string>
#include <vector>

#include "core/k_core.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

int k_core(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _OUT_ std::vector<int>& KC
) {
    int len_V = V.size() - 1;
    int len_E = E.size();

    KC = vector<int>(len_V, 0);
    int r = cuda_k_core(V.data(), E.data(), len_V, len_E, KC.data());

    return r;
}

} // namespace gpu_easygraph