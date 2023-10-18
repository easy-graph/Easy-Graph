#pragma once

#include "common.h"

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
 *     CC -
 *         closeness centrality, if sources[i] == EG_GPU_NODE_ACTIVE, CC[i] will
 *         be calculated
 * 
 * return:
 *     EG_GPU_STATUS_CODE
 */
int cuda_closeness_centrality (
    _IN_ int32_t* V, 
    _IN_ int32_t* E, 
    _IN_ float* W, 
    _IN_ int32_t* sources,
    _IN_ int32_t len_V, 
    _IN_ int32_t len_E,  
    _OUT_ float* CC
);