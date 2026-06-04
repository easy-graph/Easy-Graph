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

    m.def("cpp_degree_centrality", &degree_centrality, py::arg("G"));
    m.def("cpp_in_degree_centrality", &in_degree_centrality, py::arg("G"));
    m.def("cpp_out_degree_centrality", &out_degree_centrality, py::arg("G"));
    m.def("cpp_closeness_centrality", &closeness_centrality, py::arg("G"), py::arg("weight") = "weight", py::arg("cutoff") = py::none(), py::arg("sources") = py::none());
    m.def("cpp_betweenness_centrality", &betweenness_centrality, py::arg("G"), py::arg("weight") = "weight", py::arg("cutoff") = py::none(),py::arg("sources") = py::none(), py::arg("normalized") = py::bool_(true), py::arg("endpoints") = py::bool_(false));
    m.def("cpp_katz_centrality", &cpp_katz_centrality, py::arg("G"), py::arg("alpha") = 0.1, py::arg("beta") = 1.0, py::arg("max_iter") = 1000, py::arg("tol") = 1e-6, py::arg("normalized") = true);
    m.def("cpp_eigenvector_centrality", &cpp_eigenvector_centrality, py::arg("G"), py::arg("max_iter") = 100, py::arg("tol") = 1.0e-6, py::arg("nstart") = py::none(), py::arg("weight") = "weight");
    m.def("cpp_k_core", &core_decomposition, py::arg("G"));
    m.def("cpp_density", &density, py::arg("G"));
    m.def("cpp_constraint", &constraint, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none(), py::arg("n_workers") = py::none());
    m.def("cpp_effective_size", &effective_size, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none(), py::arg("n_workers") = py::none());
    m.def("cpp_efficiency", &efficiency, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none(), py::arg("n_workers") = py::none());
    m.def("cpp_hierarchy", &hierarchy, py::arg("G"), py::arg("nodes") = py::none(), py::arg("weight") = py::none(), py::arg("n_workers") = py::none());
    m.def("cpp_pagerank", &_pagerank, py::arg("G"), py::arg("alpha") = 0.85, py::arg("max_iterator") = 500, py::arg("threshold") = 1e-6, py::arg("weight") = "weight");
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
    m.def("cpp_boruvka_mst_edges", &boruvka_mst_edges, py::arg("G"), py::arg("minimum") = true, py::arg("weight") = "weight", py::arg("data") = true, py::arg("ignore_nan") = false);
    m.def("cpp_average_shortest_path_length", &average_shortest_path_length, py::arg("G"), py::arg("weight") = py::none(), py::arg("method") = py::none());
    m.def("cpp_eccentricity", &eccentricity, py::arg("G"), py::arg("v") = py::none(), py::arg("sp") = py::none());
    m.def("cpp_connected_components_undirected", &connected_component_undirected, py::arg("G"));
    m.def("cpp_connected_components_directed", &connected_component_directed, py::arg("G"));

    // community methods
    m.def("cpp_modularity", &cpp_modularity, py::arg("G"), py::arg("communities"), py::arg("weight") = py::str("weight"));
    m.def("cpp_greedy_modularity_communities", &cpp_greedy_modularity_communities, py::arg("G"), py::arg("weight") = py::str("weight"));
    m.def("cpp_enumerate_subgraph", &cpp_enumerate_subgraph, py::arg("G"), py::arg("k"));
    m.def("cpp_random_enumerate_subgraph", &cpp_random_enumerate_subgraph, py::arg("G"), py::arg("k"), py::arg("cut_prob"));
    m.def("cpp_louvain_communities", &cpp_louvain_communities, py::arg("G"), py::arg("weight") = py::str("weight"), py::arg("threshold") = py::float_(0.00002), py::arg("resolution") = py::float_(1.0));
    m.def("cpp_louvain_communities_serial", &cpp_louvain_communities_serial, py::arg("G"), py::arg("weight") = py::str("weight"), py::arg("threshold") = py::float_(0.00002), py::arg("resolution") = py::float_(1.0));
    m.def("cpp_LPA", &cpp_LPA, py::arg("G"));
    m.def("cpp_ego_graph", &cpp_ego_graph, py::arg("G"), py::arg("n"), py::arg("radius") = py::int_(1), py::arg("center") = py::bool_(true), py::arg("undirected") = py::bool_(false), py::arg("distance") = py::none());
    m.def("cpp_ego_graph_csr", &cpp_ego_graph_csr, py::arg("G"), py::arg("n"), py::arg("radius") = py::int_(1), py::arg("center") = py::bool_(true), py::arg("undirected") = py::bool_(false), py::arg("distance") = py::none());
    m.def("cpp_localsearch", &cpp_localsearch, py::arg("G"), py::arg("center_num") = py::none(), py::arg("auto_choose_centers") = py::bool_(false), py::arg("maximum_tree") = py::bool_(true), py::arg("seed") = py::none(), py::arg("self_loop") = py::bool_(false));
}