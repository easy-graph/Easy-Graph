#include <exception>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "common.h"

namespace py = pybind11;

using std::pair;
using std::runtime_error;
using std::string;
using std::vector;


int eg_graph_to_CSR (
    _IN_ py::object py_G,
    _IN_ py::object py_attr_weight_name,
    _OUT_ vector<int32_t>& V, 
    _OUT_ vector<pair<int, float>>& E_and_W
)
{
    py::dict edge = py_G.attr("adj");
    py::dict nodes = py_G.attr("nodes");
    py::dict node_to_index = py_G.attr("node_index");
    py::list index_to_node(node_to_index.size());
    
    for (auto it = node_to_index.begin(); it != node_to_index.end(); ++it) {
        index_to_node[it->second.cast<int>()] = it->first;
    }

    int len_E = 0;
    for (auto it = edge.begin(); it != edge.end(); ++it) {
        len_E += it->second.cast<py::dict>().size();
    }

    V = vector<int>(nodes.size(), 0);
    int range_start = 0;
    
    for (int i = 0; i < index_to_node.size(); ++i) {
        auto adjs = edge[index_to_node[i]].cast<py::dict>();
        for (auto it = adjs.begin(); it != adjs.end(); ++it) {
            int adj_idx = node_to_index[it->first].cast<int>();
            float w = 1.0;
            if (it->second.contains(py_attr_weight_name)) {
                w = it->second[py_attr_weight_name].cast<float>();
            }

            E_and_W.push_back({adj_idx, w});
        }

        int range_end = E_and_W.size();

        sort(E_and_W.begin() + range_start, E_and_W.end(), 
                [] (const pair<int, float>& p1, const pair<int, float>& p2) {
            return p1.first < p2.first;
        });
        V[i] = range_start;

        range_start = range_end;
    }
    
    return EG_GPU_SUCC;
}



int indexed_value_to_eg_node_dic (
    _IN_ py::object py_G,
    _IN_ const vector<float>& val,
    _IN_ const vector<int32_t>& sources,
    _OUT_ py::dict& node_dic
)
{
    py::dict node_to_index = py_G.attr("node_index");
    py::list index_to_node(node_to_index.size());
    
    for (auto it = node_to_index.begin(); it != node_to_index.end(); ++it) {
        index_to_node[it->second.cast<int>()] = it->first;
    }

    node_dic.clear();

    for (int i = 0; i < val.size(); ++i) {
        if (sources[i] == EG_GPU_NODE_ACTIVE) {
            node_dic[index_to_node[i]] = py::float_(val[i]);
        }
    }

    return EG_GPU_SUCC;
}


int sources_stdlize (
    _IN_ py::object py_sources,
    _IN_ int32_t len_V,
    _OUT_ vector<int32_t>& sources
)
{
    if (py_sources.is_none()) {
        sources = vector<int32_t>(len_V, EG_GPU_NODE_ACTIVE);
        return EG_GPU_SUCC;
    }

    sources = vector<int32_t>(len_V, EG_GPU_NODE_INACTIVE);
    for (auto it = py_sources.begin(); it != py_sources.end(); ++it) {
        sources[it->cast<int>()] = EG_GPU_NODE_ACTIVE;
    }

    return EG_GPU_SUCC;
}

void throw_exception (
    _IN_ int status
)
{
    // whatever input this function receives,
    // it will definitely throw an exception
    switch (status) {
        case EG_GPU_FAILED_TO_ALLOCATE_HOST_MEM:
            throw runtime_error("EasyGraph GPU: failed to allocate host mem");
            break;
        case EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM:
            throw runtime_error("EasyGraph GPU: failed to allocate gpu mem");
            break;
        case EG_GPU_DEVICE_ERR:
            throw runtime_error("EasyGraph GPU: gpu error occurred");
            break;
        case EG_GPU_UNKNOW_ERROR:
            throw runtime_error("EasyGraph GPU: gpu unkonw error");
            break;
        default:
            throw runtime_error("EasyGraph GPU: unknow error occurred");
            break;
    }
}