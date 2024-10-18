#pragma once

#include "common.h"

namespace gpu_easygraph {

int cuda_constraint(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ double* constraint_results
);

} // namespace gpu_easygraph