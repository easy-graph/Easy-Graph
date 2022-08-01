#include "Graph.h"
#include "Utils.h"
#include "Structural_holes.h"
#include "Path.h"
#include "Components.h"
#include "Not_sorted.h"

BOOST_PYTHON_MODULE(cpp_easygraph)
{

    py::class_<Graph>("Graph", py::no_init)
        .def("__init__", py::raw_function(&__init__))
        .def(py::init<>())
        .def("__iter__", &__iter__)
        .def("__len__", &__len__)
        .def("__contains__", &__contains__, (py::arg("node")))
        .def("__getitem__", &__getitem__, (py::arg("node")))
        .def("add_node", py::raw_function(&add_node))
        .def("add_nodes", &add_nodes, (py::arg("nodes_for_adding"), py::arg("nodes_attr") = py::list()))
        .def("add_nodes_from", py::raw_function(&add_nodes_from))
        .def("remove_node", &remove_node, (py::arg("node_to_remove")))
        .def("remove_nodes", &remove_nodes, (py::arg("nodes_to_remove")))
        .def("number_of_nodes", &number_of_nodes)
        .def("has_node", &has_node, (py::arg("node")))
        .def("nbunch_iter", &nbunch_iter, (py::arg("nbunch") = py::object()))
        .def("add_edge", py::raw_function(&add_edge))
        .def("add_edges", &add_edges, (py::arg("edges_for_adding"), py::arg("edges_attr") = py::list()))
        .def("add_edges_from", py::raw_function(&add_edges_from))
        .def("add_edges_from_file", &add_edges_from_file, (py::arg("file"), py::arg("weighted") = false))
        .def("add_weighted_edge", &add_weighted_edge, (py::arg("u_of_edge"), py::arg("v_of_edge"), py::arg("weight")))
        .def("remove_edge", &remove_edge, (py::arg("u"), py::arg("v")))
        .def("remove_edges", &remove_edges, (py::arg("edges_to_remove")))
        .def("number_of_edges", &number_of_edges, (py::arg("u") = py::object(), py::arg("v") = py::object()))
        .def("has_edge", &has_edge, (py::arg("u"), py::arg("y")))
        .def("copy", &copy)
        .def("degree", &degree, (py::arg("weight") = py::object("weight")))
        .def("neighbors", &neighbors, (py::arg("node")))
        .def("all_neighbors", &neighbors, (py::arg("node")))
        .def("nodes_subgraph", &nodes_subgraph, (py::arg("from_nodes")))
        .def("ego_subgraph", &ego_subgraph, (py::arg("center")))
        .def("size", &size, (py::arg("weight") = py::object()))
        .def("is_directed", &is_directed)
        .def("is_multigraph", &is_multigraph)
        .add_property("graph", &Graph::get_graph)
        .add_property("nodes", &Graph::get_nodes)
        .add_property("name", &Graph::get_name)
        .add_property("adj", &Graph::get_adj)
        .add_property("edges", &Graph::get_edges);
    
    py::def("cpp_constraint", &constraint, (py::arg("G"), py::arg("nodes") = py::object(), py::arg("weight") = py::object(), py::arg("n_workers") = py::object()));
    py::def("cpp_effective_size", &effective_size, (py::arg("G"), py::arg("nodes") = py::object(), py::arg("weight") = py::object(), py::arg("n_workers") = py::object()));
    py::def("cpp_hierarchy", &hierarchy, (py::arg("G"), py::arg("nodes") = py::object(), py::arg("weight") = py::object(), py::arg("n_workers") = py::object()));
    py::def("cpp_dijkstra_multisource", &_dijkstra_multisource, (py::arg("G"), py::arg("sources"), py::arg("weight") = "weight", py::arg("target") = py::object()));
    py::def("cpp_clustering", &clustering, (py::arg("G"), py::arg("nodes") = py::object(), py::arg("weight") = py::object()));
    py::def("cpp_Floyd", &Floyd, (py::arg("G")));
    py::def("cpp_Prim", &Prim, (py::arg("G")));
    py::def("cpp_Kruskal", &Kruskal, (py::arg("G")));
}