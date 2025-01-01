#pragma once

#include "common.h"

namespace gpu_easygraph {

int cuda_constraint(
    _IN_ int num_nodes,
    _IN_ int len_rowPtrOut,
    _IN_ int len_colIdxOut,
    _IN_ const int* rowPtrOut,
    _IN_ const int* colIdxOut,
    _IN_ const double* valOut,
    _IN_ const int* rowPtrIn,
    _IN_ const int* colIdxIn,
    _IN_ const double* valIn,
    _IN_ bool is_directed,
    _IN_ int* node_mask,
    _OUT_ double* constraints
);

} // namespace gpu_easygraph