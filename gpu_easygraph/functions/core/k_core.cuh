# pragma once

#include "common.h"

namespace gpu_easygraph {

/**
 * description: 
 *     use cuda to calculate k core. the graph must be 
 *     in CSR format.
 * 
 * arguments:
 *     V -
 *         the vertices in CSR format
 * 
 *     E -
 *         the edges in CSR format
 * 
 *     len_V -
 *         len of V
 * 
 *     len_E -
 *         len of E
 * 
 *     k_core_res -
 *         result of k_core
 * 
 * return:
 *     EG_GPU_STATUS_CODE
 */
int cuda_k_core (
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* k_core_res
);

} // namespace gpu_easygraph