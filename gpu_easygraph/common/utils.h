#pragma once

#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "common.h"

/**
 * description: 
 *     convert a easygraph python format graph to 
 *     compressed sparse row(CSR) format
 * 
 * arguments:
 *     py_attr_weight_name -
 *         the attribute name of edge weight in edge attributes dict
 * 
 * return:
 *     EG_GPU_STATUS_CODE
 */
int eg_graph_to_CSR (
    _IN_ pybind11::object py_G,
    _IN_ pybind11::object py_attr_weight_name,
    _OUT_ std::vector<int>& V, 
    _OUT_ std::vector<std::pair<int, double>>& E_and_W
);




/**
 * description: 
 *     Fill in the indeces of nodes needed to calculate CC
 * 
 * arguments:
 *     py_sources -
 *         sources containing types of nodes
 * 
 *     sources -
 *         sources containing indeces of nodes
 * 
 * return:
 *     EG_GPU_STATUS_CODE
 */
int sources_stdlize (
    _IN_ pybind11::object py_G,
    _IN_ pybind11::object py_sources,
    _IN_ int len_V,
    _OUT_ std::vector<int>& sources
);



/**
 * description: 
 *     decide the GPU-ver BC parameter "warp_size" by len_V and len_E
 * 
 * arguments:
 *     len_V -
 *         size of vertices
 * 
 *     len_E -
 *         size of edges
 * 
 * return:
 *     a proper warp_size
 */
int decide_warp_size (
    _IN_ int len_V,
    _IN_ int len_E
);




/**
 * description: 
 *     will definitely throw an exception with whatever 'status' it receives
 * 
 * arguments:
 *     status -
 *         EG_GPU_STATUS_CODE
 */
void throw_exception (
    _IN_ int status
);