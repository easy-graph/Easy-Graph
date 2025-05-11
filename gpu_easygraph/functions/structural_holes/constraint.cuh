#pragma once

#include "common.h"

namespace gpu_easygraph {

int cuda_constraint(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const int* row,
    _IN_ const int* col,
    _IN_ const double* W,
    _IN_ int num_nodes,
    _IN_ int num_edges,
    _IN_ bool is_directed,
    _IN_ int* node_mask,
    _OUT_ double* constraint_results
);

} // namespace gpu_easygraph