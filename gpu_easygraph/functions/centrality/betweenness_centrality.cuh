#pragma once

#include "common.h"

namespace gpu_easygraph {

/**
 * description: 
 *     use cuda to calculate betweenness_centrality. the graph must be 
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
 *         set of source vertices to consider when calculating shortest paths.
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
 *     is_directed -
 *         if this graph is directed
 * 
 *     normalized -
 *         if the answer needs to be normalized
 * 
 *     endpoints -
 *         if true include the endpoints in the shortest basic counts.
 * 
 *     BC -
 *         betweenness centrality output
 * 
 * return:
 *     EG_GPU_STATUS_CODE
 */
int cuda_betweenness_centrality (
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _IN_ int is_directed,
    _IN_ int normalized,
    _IN_ int endpoints,
    _OUT_ double* BC
);

} // namespace gpu_easygraph