#include "centrality/centrality.h"
#include "centrality/closeness_centrality.cuh"
#include "common.h"
#include "utils.h"

namespace py = pybind11;

using std::pair;
using std::string;
using std::vector;

py::object closeness_centrality(
    _IN_ py::object py_G, 
    _IN_ py::object py_weight, 
    _OUT_ py::object py_sources
)
{
    vector<int32_t> V;
    vector<pair<int, float>> E_and_W;

    eg_graph_to_CSR(py_G, py_weight, V, E_and_W);

    vector<int32_t> sources;
    sources_stdlize(py_sources, V.size(), sources);

    vector<int32_t> E(E_and_W.size());
    vector<float> W(E_and_W.size());
    for (int i = 0; i < E_and_W.size(); ++i) {
        E[i] = E_and_W[i].first;
        W[i] = E_and_W[i].second;
    }

    vector<float> CC(V.size());

    int r = cuda_closeness_centrality(V.data(), E.data(), W.data(),
            sources.data(), V.size(), E.size(), CC.data());

    if (r != EG_GPU_SUCC) {
        // meets an error, throw an exception
        throw_exception(r);
        // the following codes will never be executed
    }

    py::dict ret;
    indexed_value_to_eg_node_dic(py_G, CC, sources, ret);
    
    return ret;
}