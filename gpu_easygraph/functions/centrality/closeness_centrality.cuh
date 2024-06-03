#pragma once

#include "common.h"

namespace gpu_easygraph {

/**
 * description: 
 *     use cuda to calculate closeness_centrality. the graph must be 
 *     in CSR format.
 * 
 * arguments:
 *     V -
 *         the vertices in CSR format
 * 
 *     E -
 *         the edges in CSR format
 * 
 *     W -
 *         the weight of edges in CSR format
 * 
 *     sources -
 *         an array of EG_GPU_NODE_STATUS. the according CC[i] will be 
 *         calculated only if sources[i] == EG_GPU_NODE_ACTIVE
 * 
 *     len_V -
 *         len of V
 * 
 *     len_E -
 *         len of E
 * 
 *     warp_size -
 *         the number of threads assigned to a vertex
 * 
 *     CC -
 *         closeness centrality output
 * 
 * return:
 *     EG_GPU_STATUS_CODE
 */
int cuda_closeness_centrality (
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _OUT_ double* CC
);

} // namespace gpu_easygraph