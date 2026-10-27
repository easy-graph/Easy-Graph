#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "algorithms/pagerank.h"
#include "communicator.h"
#include "distributed_graph.h"
#include "loader.h"
#include "../common/pagerank_parallel.h"

namespace py = pybind11;
using namespace egdist;

PYBIND11_MODULE(cpp_easygraph_distributed, m) {
    m.doc() = "Optional MPI backend for EasyGraph distributed analytics";
    m.def("finalize_mpi", &finalize_mpi);
    m.def("configure_parallelism", &easygraph_parallel::configure, py::arg("threads"));
    m.def("parallel_info", []() {
        py::dict info;
        info["openmp_enabled"] = easygraph_parallel::enabled();
        info["max_threads"] = easygraph_parallel::max_threads();
        info["team_size"] = easygraph_parallel::team_size();
        return info;
    });
    py::class_<Communicator, std::shared_ptr<Communicator>>(m, "Communicator")
        .def(py::init<>())
        .def_property_readonly("rank", &Communicator::rank)
        .def_property_readonly("size", &Communicator::size)
        .def_property_readonly("closed", &Communicator::closed)
        .def("close", &Communicator::close)
        .def("barrier", &Communicator::barrier)
        .def("check_error", &Communicator::check_error)
        .def("agree", &Communicator::agree)
        .def("broadcast_string", &Communicator::broadcast_string)
        .def("gather_result", &Communicator::gather_result);
    py::class_<DistributedGraph, std::shared_ptr<DistributedGraph>>(m, "DistributedGraph")
        .def_readonly("vertices", &DistributedGraph::vertices)
        .def_readonly("global_num_vertices", &DistributedGraph::global_num_vertices)
        .def_readonly("global_num_edges", &DistributedGraph::global_num_edges)
        .def_property_readonly("local_num_arcs", [](const DistributedGraph& g) { return g.targets.size(); });
    m.def("read_edgelist", &read_edgelist, py::arg("path"), py::arg("comm"), py::arg("directed"));
    m.def("pagerank", [](const DistributedGraph& graph, double alpha, int max_iter,
                          double tol, bool weighted) {
        auto result = pagerank(graph, alpha, max_iter, tol, weighted);
        return py::make_tuple(result.ids, result.values, result.iterations);
    }, py::arg("graph"), py::arg("alpha"), py::arg("max_iter"), py::arg("tol"), py::arg("weighted"));
}
