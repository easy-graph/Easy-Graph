#include <pybind11/pybind11.h>

#include "centrality/centrality.h"

namespace py = pybind11;

PYBIND11_MODULE(gpu_easygraph, m) {
    m.def("gpu_closeness_centrality", &closeness_centrality, py::arg("G"), py::arg("weight") = "weight", py::arg("sources") = py::none());
}