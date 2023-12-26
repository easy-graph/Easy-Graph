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
    _OUT_ vector<int>& V, 
    _OUT_ vector<pair<int, double>>& E_and_W
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
            double w = 1.0;
            if (it->second.contains(py_attr_weight_name)) {
                w = it->second[py_attr_weight_name].cast<double>();
            }

            E_and_W.push_back({adj_idx, w});
        }

        int range_end = E_and_W.size();

        sort(E_and_W.begin() + range_start, E_and_W.end(), 
                [] (const pair<int, double>& p1, const pair<int, double>& p2) {
            return p1.first < p2.first;
        });
        V[i] = range_start;

        range_start = range_end;
    }
    
    return EG_GPU_SUCC;
}




int sources_stdlize (
    _IN_ py::object py_G,
    _IN_ py::object py_sources,
    _IN_ int len_V,
    _OUT_ vector<int>& sources
)
{
    py::dict node_to_index = py_G.attr("node_index");

    sources.clear();
    if (py_sources.is_none()) {
        for (int i = 0; i < len_V; ++i) {
            sources.push_back(i);
        }
        return EG_GPU_SUCC;
    }

    for (auto it = py_sources.begin(); it != py_sources.end(); ++it) {
        sources.push_back(node_to_index[*it].cast<int>());
    }

    return EG_GPU_SUCC;
}



int decide_warp_size (
    _IN_ int len_V,
    _IN_ int len_E
)
{
    vector<int> warp_size_cand{1, 2, 4, 8, 16, 32};

    if (len_E / len_V < warp_size_cand.front()) {
        return warp_size_cand.front();
    }

    for (int i = 0; i + 1 < warp_size_cand.size(); ++i) {
        if (warp_size_cand[i] <= len_E / len_V
                && len_E / len_V < warp_size_cand[i + 1]) {
            return warp_size_cand[i + 1];
        }
    }
    return warp_size_cand.back();
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