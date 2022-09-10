#include "connected.h"

#include <pybind11/stl.h>

#include "../../classes/graph.h"
#include "../../common/utils.h"

std::set<node_t> plain_bfs(py::object G, py::object source) {
    Graph& G_ = G.cast<Graph&>();
    node_t source_id = G_.node_to_id.attr("get")(source).cast<node_t>();
    adj_dict_factory& G_adj = G_.adj;
    std::set<node_t> seen;
    std::set<node_t> nextlevel;
    nextlevel.emplace(source_id);
    std::set<node_t> res;
    while (nextlevel.size()) {
        std::set<node_t> thislevel = nextlevel;
        nextlevel = std::set<node_t>();
        for (std::set<node_t>::iterator i = thislevel.begin(); i != thislevel.end(); i++) {
            node_t v_id = *i;
            if (seen.find(v_id) == seen.end()) {
                res.emplace(v_id);
                seen.emplace(v_id);
                adj_attr_dict_factory v_adj = G_adj[v_id];
                for (adj_attr_dict_factory::iterator j = v_adj.begin(); j != v_adj.end(); j++) {
                    node_t neighbor_id = j->first;
                    if (seen.find(neighbor_id) == seen.end()) {
                        nextlevel.emplace(neighbor_id);
                    }
                }
            }
        }
    }
    return res;
}

std::vector<std::set<node_t>> generator_connected_components(py::object G) {
    Graph& G_ = G.cast<Graph&>();
    std::unordered_set<node_t> seen;
    std::vector<std::set<node_t>> component_res;
    std::set<std::set<node_t>> temp_res;
    node_dict_factory nodes_list = G_.node;
    for (node_dict_factory::iterator iter = nodes_list.begin(); iter != nodes_list.end(); iter++) {
        node_t node_id = iter->first;
        if (seen.find(node_id) == seen.end()) {
            py::object node = G_.id_to_node.attr("get")(node_id);
            std::set<node_t> component = plain_bfs(G, node);
            temp_res.emplace(component);
            for (std::set<node_t>::iterator j = component.begin(); j != component.end(); j++) {
                seen.emplace(*j);
            }
        }
    }
    for (std::set<std::set<node_t>>::iterator iter = temp_res.begin(); iter != temp_res.end(); iter++) {
        component_res.emplace_back(*iter);
    }
    return component_res;
}
