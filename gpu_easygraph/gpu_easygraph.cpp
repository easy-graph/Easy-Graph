#include <pybind11/pybind11.h>

#include "centrality/centrality.h"
#include "core/core.h"

namespace py = pybind11;

PYBIND11_MODULE(gpu_easygraph, m) {
    m.def("gpu_closeness_centrality", 
        &closeness_centrality, 
        py::arg("G"), 
        py::arg("weight") = py::none(), 
        py::arg("sources") = py::none()
    );
    m.def("gpu_betweenness_centrality", 
        &betweenness_centrality, 
        py::arg("G"), 
        py::arg("weight") = py::none(), 
        py::arg("sources") = py::none(),
        py::arg("normalized") = py::bool_(true),
        py::arg("endpoints") = py::bool_(false)
    );
    m.def("gpu_k_core", 
        &k_core, 
        py::arg("G")
    );
}