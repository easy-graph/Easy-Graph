#include "core/core.h"
#include "core/k_core.cuh"
#include "common.h"
#include "utils.h"

#include <chrono>//TMPTMPTMP 123123123

namespace py = pybind11;

using std::string;
using std::vector;
using std::pair;

py::object k_core(
    _IN_ py::object py_G
)
{
int64_t t1 = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
    vector<int> V;
    vector<int> E;

    eg_graph_to_CSR(py_G, V, E);


    int len_V = V.size() - 1;
    int len_E = E.size();

    vector<int> k_core_res(len_V, 0);
int64_t t2 = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
    int r = cuda_k_core(V.data(), E.data(), len_V, len_E, k_core_res.data());
int64_t t3 = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count();

    if (r != EG_GPU_SUCC) {
        throw_exception(r);
    }

    py::list ret;
    for (int i = 0; i < k_core_res.size(); ++i) {
        ret.append(k_core_res[i]);
    }
int64_t t4 = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
printf("t2-t1: %ld ; t3-t2: %ld ; t4-t3: %ld\n", t2 - t1, t3 - t2, t4 - t3);
    
    return ret;
}
