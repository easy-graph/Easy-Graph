#include "centrality/centrality.h"
#include "centrality/closeness_centrality.cuh"
#include "centrality/betweenness_centrality.cuh"
#include "common.h"
#include "utils.h"

namespace py = pybind11;

using std::pair;
using std::string;
using std::vector;

py::object closeness_centrality(
    _IN_ py::object py_G, 
    _IN_ py::object py_weight, 
    _IN_ py::object py_sources
)
{
    vector<int> V;
    vector<pair<int, double>> E_and_W;

    eg_graph_to_CSR(py_G, py_weight, V, E_and_W);

    vector<int> E(E_and_W.size());
    vector<double> W(E_and_W.size());
    for (int i = 0; i < E_and_W.size(); ++i) {
        E[i] = E_and_W[i].first;
        W[i] = E_and_W[i].second;
    }

    int len_V = V.size() - 1;
    int len_E = E.size();

    vector<int> sources;
    sources_stdlize(py_G, py_sources, len_V, sources);

    int warp_size = decide_warp_size(len_V, len_E);

    vector<double> CC(len_V);

    int r = cuda_closeness_centrality(V.data(), E.data(), W.data(), 
            sources.data(), len_V, len_E, sources.size(), 
            warp_size, CC.data());

    if (r != EG_GPU_SUCC) {
        throw_exception(r);
    }

    py::list ret;
    for (int i = 0; i < sources.size(); ++i) {
        ret.append(CC[i]);
    }
    
    return ret;
}

py::object betweenness_centrality(
    _IN_ py::object py_G, 
    _IN_ py::object py_weight, 
    _IN_ py::object py_sources,
    _IN_ py::object py_normalized,
    _IN_ py::object py_endpoints
)
{
    vector<int> V;
    vector<pair<int, double>> E_and_W;

    eg_graph_to_CSR(py_G, py_weight, V, E_and_W);

    vector<int> E(E_and_W.size());
    vector<double> W(E_and_W.size());
    for (int i = 0; i < E_and_W.size(); ++i) {
        E[i] = E_and_W[i].first;
        W[i] = E_and_W[i].second;
    }

    int len_V = V.size() - 1;
    int len_E = E.size();

    vector<int> sources;
    sources_stdlize(py_G, py_sources, len_V, sources);

    int warp_size = decide_warp_size(len_V, len_E);
    int normalized = py_normalized.cast<py::bool_>();
    int endpoints = py_endpoints.cast<py::bool_>();
    int is_directed = py_G.attr("is_directed")().cast<py::bool_>();

    vector<double> BC(len_V);

    int r = cuda_betweenness_centrality(V.data(), E.data(), W.data(),
            sources.data(), len_V, len_E, sources.size(),
            warp_size, is_directed, normalized, endpoints, BC.data());

    if (r != EG_GPU_SUCC) {
        throw_exception(r);
    }

    py::list ret;
    for (int i = 0; i < BC.size(); ++i) {
        ret.append(BC[i]);
    }
    
    return ret;
}
