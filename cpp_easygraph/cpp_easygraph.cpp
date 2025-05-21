#include "classes/__init__.h"
#include "functions/__init__.h"
PYBIND11_MODULE(cpp_easygraph, m) {

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("__init__", &Graph__init__)
        .def("__iter__", &Graph__iter__)
        .def("__len__", &Graph__len__)
        .def("__contains__", &Graph__contains__, py::arg("node"))
        .def("__getitem__", &Graph__getitem__, py::arg("node"))
        .def("add_node", &Graph_add_node)
        .def("add_nodes", &Graph_add_nodes, py::arg("nodes_for_adding"), py::arg("nodes_attr") = py::list())
        .def("add_nodes_from", &Graph_add_nodes_from)
        .def("remove_node", &Graph_remove_node, py::arg("node_to_remove"))
        .def("remove_nodes", &Graph_remove_nodes, py::arg("nodes_to_remove"))
        .def("number_of_nodes", &Graph_number_of_nodes)
        .def("has_node", &Graph_has_node, py::arg("node"))
        .def("nbunch_iter", &Graph_nbunch_iter, py::arg("nbunch") = py::none())
        .def("add_edge", &Graph_add_edge)
        .def("add_edges", &Graph_add_edges, py::arg("edges_for_adding"), py::arg("edges_attr") = py::list())
        .def("add_edges_from", &Graph_add_edges_from)
        .def("add_edges_from_file", &Graph_add_edges_from_file, py::arg("file"), py::arg("weighted") = false, py::arg("is_transform") = false)
        .def("add_weighted_edge", &Graph_add_weighted_edge, py::arg("u_of_edge"), py::arg("v_of_edge"), py::arg("weight"))
        .def("remove_edge", &Graph_remove_edge, py::arg("u"), py::arg("v"))
        .def("remove_edges", &Graph_remove_edges, py::arg("edges_to_remove"))
        .def("number_of_edges", &Graph_number_of_edges, py::arg("u") = py::none(), py::arg("v") = py::none())
        .def("has_edge", &Graph_has_edge, py::arg("u"), py::arg("y"))
        .def("copy", &Graph_copy)
        .def("degree", &Graph_degree, py::arg("weight") = "weight")
        .def("neighbors", &Graph_neighbors, py::arg("node"))
        .def("all_neighbors", &Graph_neighbors, py::arg("node"))
        .def("nodes_subgraph", &Graph_nodes_subgraph, py::arg("from_nodes"))
        .def("ego_subgraph", &Graph_ego_subgraph, py::arg("center"))
        .def("size", &Graph_size, py::arg("weight") = py::none())
        .def("is_directed", &Graph_is_directed)
        .def("is_multigraph", &Graph_is_multigraph)
        .def("to_index_node_graph", &Graph_to_index_node_graph, py::arg("begin_index") = 0)
        .def("py", &Graph_py)
        .def_property("graph", &Graph::get_graph, nullptr)
        .def_property("nodes", &Graph::get_nodes, nullptr)
        .def_property("name", &Graph::get_name, &Graph::set_name)
        .def_property("adj", &Graph::get_adj, nullptr)
        .def_property("edges", &Graph::get_edges, nullptr)
        .def_property("node_index", &Graph::get_node_index, nullptr)
        .def("generate_linkgraph", &Graph_generate_linkgraph,py::arg("weight") = "weight");

    py::class_<DiGraph, Graph>(m, "DiGraph")
        .def(py::init<>())
        .def("__init__", &DiGraph__init__)
        .def("out_degree", &DiGraph_out_degree, py::arg("weight") = "weight")
        .def("in_degree", &DiGraph_in_degree, py::arg("weight") = "weight")
        .def("degree", &DiGraph_degree, py::arg("weight") = "weight")
        .def("size", &DiGraph_size, py::arg("weight") = py::none())
        .def("number_of_edges", &DiGraph_number_of_edges, py::arg("u") = py::none(), py::arg("v") = py::none())
        .def("successors", &Graph_neighbors, py::arg("node"))
        .def("predecessors", &DiGraph_predecessors, py::arg("node"))
        .def("add_node", &DiGraph_add_node)
        .def("add_nodes", &DiGraph_add_nodes, py::arg("nodes_for_adding"), py::arg("nodes_attr") = py::list())
        .def("add_nodes_from", &DiGraph_add_nodes_from)
        .def("remove_node", &DiGraph_remove_node, py::arg("node_to_remove"))
        .def("remove_nodes", &DiGraph_remove_nodes, (py::arg("nodes_to_remove")))
        .def("add_edge", &DiGraph_add_edge)
        .def("add_edges", &DiGraph_add_edges, py::arg("edges_for_adding"), py::arg("edges_attr") = py::list())
        .def("add_edges_from", &DiGraph_add_edges_from)
        .def("add_edges_from_file", &DiGraph_add_edges_from_file, py::arg("file"), py::arg("weighted") = false, py::arg("is_transform") = false)
        .def("add_weighted_edge", &DiGraph_add_weighted_edge, py::arg("u_of_edge"), py::arg("v_of_edge"), py::arg("weight"))
        .def("remove_edge", &DiGraph_remove_edge, py::arg("u"), py::arg("v"))
        .def("remove_edges", &DiGraph_remove_edges, py::arg("edges_to_remove"))
        .def("remove_edges_from", &DiGraph_remove_edges_from, py::arg("ebunch"))
        .def("nodes_subgraph", &DiGraph_nodes_subgraph, py::arg("from_nodes"))
        .def("is_directed", &DiGraph_is_directed)
        .def("py", &DiGraph_py)
        .def_property("edges", &DiGraph::get_edges, nullptr)
        .def_property("pred", &DiGraph::get_pred,nullptr)
        .def("generate_linkgraph", &DiGraph_generate_linkgraph,py::arg("weight") = "weight");

    m.def("cpp_closeness_centrality", &closeness_centrality, py::arg("G"), py::arg("weight") = "weight", py::arg("cutoff") = py::none(), py::arg("sources") = py::none());
    m.def("cpp_betweenness_centrality", &betweenness_centrality, py::arg("G"), py::arg("weight") = "weight", py::arg("cutoff") = py::none(),py::arg("sources") = py::none(), py::arg("normalized") = py::bool_(true), py::arg("endpoints") = py::bool_(false));
    m.def("cpp_k_core", &core_decomposition, py::arg("G"));
    m.def("cpp_density", &density, py::arg("G"));
    m.def("cpp_constraint", &constraint, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none(), py::arg("n_workers") = py::none());
    m.def("cpp_effective_size", &effective_size, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none(), py::arg("n_workers") = py::none());
    m.def("cpp_efficiency", &efficiency, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none(), py::arg("n_workers") = py::none());
    m.def("cpp_hierarchy", &hierarchy, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none(), py::arg("n_workers") = py::none());
    m.def("cpp_pagerank", &_pagerank, py::arg("G"), py::arg("alpha") = 0.85, py::arg("max_iterator") = 500, py::arg("threshold") = 1e-6);
    m.def("cpp_dijkstra_multisource", &_dijkstra_multisource, py::arg("G"), py::arg("sources"), py::arg("weight") = "weight", py::arg("target") = py::none());    
    m.def("cpp_spfa", &_spfa, py::arg("G"), py::arg("source"), py::arg("weight") = "weight");
    m.def("cpp_clustering", &clustering, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none());
    m.def("cpp_biconnected_dfs_record_edges", &_biconnected_dfs_record_edges, py::arg("G"), py::arg("need_components") = true);
    m.def("cpp_strongly_connected_components",&strongly_connected_components,py::arg("G"));
    m.def("cpp_Floyd", &Floyd, py::arg("G"), py::arg("weight") = "weight");
    m.def("cpp_Prim", &Prim, py::arg("G"), py::arg("weight") = "weight");
    m.def("cpp_Kruskal", &Kruskal, py::arg("G"), py::arg("weight") = "weight");
    m.def("cpp_plain_bfs", &plain_bfs, py::arg("G"), py::arg("source"));
    m.def("cpp_kruskal_mst_edges", &kruskal_mst_edges, py::arg("G"), py::arg("minimum") = true, py::arg("weight") = "weight", py::arg("data") = true, py::arg("ignore_nan") = false);
    m.def("cpp_prim_mst_edges", &prim_mst_edges, py::arg("G"), py::arg("minimum") = true, py::arg("weight") = "weight", py::arg("data") = true, py::arg("ignore_nan") = false);
    m.def("cpp_connected_components_undirected", &connected_component_undirected, py::arg("G"));
    m.def("cpp_connected_components_directed", &connected_component_directed, py::arg("G"));
}