#include "centrality.h"

#include "../../classes/graph.h"
#include "../../classes/directed_graph.h" 
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"

namespace py = pybind11;

py::object degree_centrality(
    py::object G
) {
    Graph* graph = G.cast<Graph*>();
    py::dict centrality_map = py::dict();
    py::object nodes = graph->get_nodes();
    int n = py::len(nodes);
    if (n <= 1) {
        for (const auto& node_handle : nodes) {
            centrality_map[node_handle] = 0.0;
        }
        return centrality_map;
    }

    double scale = 1.0 / (n - 1);

    std::string class_name = G.attr("__class__").attr("__name__").cast<std::string>();

    if (class_name == "DiGraphC") {
        // 有向图 (DiGraph) 
        DiGraph* digraph = G.cast<DiGraph*>();
        py::object adj = digraph->get_adj();
        py::object pred = digraph->get_pred(); 

        for (const auto& node_handle : nodes) {
            int out_deg = py::len(adj[node_handle]);
            int in_deg = py::len(pred[node_handle]);
            centrality_map[node_handle] = (double)(out_deg + in_deg) * scale;
        }
    } else {
        py::object adj = graph->get_adj();
        for (const auto& node_handle : nodes) {
            int degree = py::len(adj[node_handle]);
            centrality_map[node_handle] = (double)degree * scale;
        }
    }
    return centrality_map;
}


py::object in_degree_centrality(
    py::object G
) {
    DiGraph* graph = G.cast<DiGraph*>();
    py::dict centrality_map = py::dict();

    py::object nodes = graph->get_nodes();
    int n = py::len(nodes);

    if (n <= 1) {
        return centrality_map;
    }

    double scale = 1.0 / (n - 1);

    py::object pred = graph->get_pred(); 

    for (const auto& node_handle : nodes) {
        int in_degree = py::len(pred[node_handle]);
        centrality_map[node_handle] = in_degree * scale;
    }
    return centrality_map;
}


py::object out_degree_centrality(
    py::object G
) {
    Graph* graph = G.cast<Graph*>(); 
    py::dict centrality_map = py::dict();

    py::object nodes = graph->get_nodes();
    int n = py::len(nodes);

    if (n <= 1) {
        return centrality_map;
    }
    double scale = 1.0 / (n - 1);

    py::object adj = graph->get_adj(); 

    for (const auto& node_handle : nodes) {
        int out_degree = py::len(adj[node_handle]);
        centrality_map[node_handle] = out_degree * scale;
    }
    return centrality_map;
}