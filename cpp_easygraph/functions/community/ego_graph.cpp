#include "ego_graph.h"
#include "subgraph.h"
#include "../../classes/graph.h"
#include "../../classes/directed_graph.h"
#include "../../classes/csr_graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include "../path/path.h"
#include <algorithm>


struct _EgoGraphCore {
    Graph_L G_l;                                  
    std::vector<node_t> node_ids;                 
    std::unordered_map<node_t, int> node_to_idx;  
    py::dict id_to_node_py;                       
    bool has_error = false;                       
};


static _EgoGraphCore _cpp_ego_graph_compute_impl(
    Graph& G_,
    py::object n,
    double radius_val,
    bool center_val,
    bool undirected_val,
    bool is_directed,
    const std::string& weight_key) {

    _EgoGraphCore out;

    if (G_.node_to_id.contains(n) == 0) {
        PyErr_Format(PyExc_KeyError, "Node %R is not in the graph.", n.ptr());
        out.has_error = true;
        return out;
    }

    node_t center_id = G_.node_to_id[n].cast<node_t>();
    out.id_to_node_py = G_.id_to_node;

    if (G_.linkgraph_dirty || G_.linkgraph_structure.max_deg == -1) {
        if (undirected_val) {
            out.G_l = graph_to_linkgraph(G_, false, weight_key, true, false);
        } else {
            out.G_l = graph_to_linkgraph(G_, is_directed, weight_key, true, false);
        }
        G_.linkgraph_dirty = false;
        G_.linkgraph_structure = out.G_l;  // also cache the freshly built linkgraph
    } else {
        out.G_l = G_.linkgraph_structure;
    }

    std::vector<double> dist = _dijkstra(out.G_l, center_id, -1);
    int N = out.G_l.n;
    for (int i = 1; i <= N; i++) {
        if (dist[i] > radius_val) continue;
        if (!center_val && i == (int)center_id) continue;
        out.node_to_idx[i] = (int)out.node_ids.size();
        out.node_ids.push_back(i);
    }
    return out;
}


static _EgoGraphCore _cpp_ego_graph_compute(
    py::object G,
    py::object n,
    py::object radius,
    py::object center,
    py::object undirected,
    py::object distance) {

    bool is_directed     = G.attr("is_directed")().cast<bool>();
    bool center_val      = center.cast<bool>();
    bool undirected_val  = undirected.cast<bool>();
    double radius_val    = radius.cast<double>();
    std::string weight_key = weight_to_string(distance);

    if (is_directed) {
        DiGraph& G_ = G.cast<DiGraph&>();
        return _cpp_ego_graph_compute_impl(
            G_, n, radius_val, center_val, undirected_val, is_directed, weight_key);
    } else {
        Graph& G_ = G.cast<Graph&>();
        return _cpp_ego_graph_compute_impl(
            G_, n, radius_val, center_val, undirected_val, is_directed, weight_key);
    }
}


py::object cpp_ego_graph(
    py::object G,
    py::object n,
    py::object radius,
    py::object center,
    py::object undirected,
    py::object distance) {

    _EgoGraphCore core = _cpp_ego_graph_compute(G, n, radius, center, undirected, distance);
    if (core.has_error) return py::none();

    return nodes_subgraph_cpp(G, core.node_ids);
}


py::object cpp_ego_graph_csr(
    py::object G,
    py::object n,
    py::object radius,
    py::object center,
    py::object undirected,
    py::object distance) {

    _EgoGraphCore core = _cpp_ego_graph_compute(G, n, radius, center, undirected, distance);
    if (core.has_error) return py::none();

    int n_nodes = (int)core.node_ids.size();
    if (n_nodes == 0) {
        return py::dict();
    }

    std::vector<int> V(n_nodes + 1, 0);
    std::vector<int> E;
    std::vector<double> W;

    for (int new_u = 0; new_u < n_nodes; new_u++) {
        node_t u = core.node_ids[new_u];
        V[new_u + 1] = V[new_u];

        int edge_idx = core.G_l.head[u];
        while (edge_idx != -1 && edge_idx < core.G_l.e) {
            node_t v = core.G_l.edges[edge_idx].to;
            auto it = core.node_to_idx.find(v);
            if (it != core.node_to_idx.end()) {
                E.push_back(it->second);
                W.push_back(core.G_l.edges[edge_idx].w);
                V[new_u + 1]++;
            }
            edge_idx = core.G_l.edges[edge_idx].next;
        }
    }

    py::list original_nodes;
    for (node_t internal_id : core.node_ids) {
        original_nodes.append(core.id_to_node_py[py::cast(internal_id)]);
    }

    py::dict result;
    result["nodes"] = original_nodes;
    result["V"] = py::cast(V);
    result["E"] = py::cast(E);
    result["W"] = py::cast(W);

    return result;
}
