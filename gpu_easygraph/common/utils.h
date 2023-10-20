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
    _OUT_ std::vector<int32_t>& V, 
    _OUT_ std::vector<std::pair<int, float>>& E_and_W
);



/**
 * description: 
 *     convert the results represented by an array indexed with
 *     node_index to node_dic. The <key, value> pair of node_dic
 *     is <node name, value from results>
 * 
 * arguments:
 *     val -
 *         the results 
 * 
 *     sources -
 *         the node will be inserted into node_dic only if sources[i] is
 *         EG_GPU_NODE_ACTIVE
 * 
 *     node_dic -
 *         the key is node name represented by a hashable python object,
 *         namely index2node[i]
 *         the value is from val[i]
 * 
 * return:
 *     EG_GPU_STATUS_CODE
 */
int indexed_value_to_eg_node_dic (
    _IN_ pybind11::object py_G,
    _IN_ const std::vector<float>& val,
    _IN_ const std::vector<int32_t>& sources,
    _OUT_ pybind11::dict& node_dic
);



/**
 * description: 
 *     convert the sources represented by python list to a vector which 
 *     contains EG_GPU_NODE_STATUS
 * 
 * arguments:
 *     py_sources -
 *         sources with python list type
 * 
 *     sources -
 *         sources with std::vector format, the elements of it are either
 *         EG_GPU_NODE_ACTIVE or EG_GPU_NODE_INACTIVE
 * 
 * return:
 *     EG_GPU_STATUS_CODE
 */
int sources_stdlize (
    _IN_ pybind11::object py_sources,
    _IN_ int32_t len_V,
    _OUT_ std::vector<int32_t>& sources
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