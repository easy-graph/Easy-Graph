#include "connected.h"

#include <pybind11/stl.h>

#include "../../classes/graph.h"
#include "../../common/utils.h"

py::object plain_bfs(py::object G, py::object source) {
    Graph& G_ = G.cast<Graph&>();
    node_t source_id = G_.node_to_id.attr("get")(source).cast<node_t>();
    adj_dict_factory& G_adj = G_.adj;
    std::unordered_set<node_t> seen;
    std::unordered_set<node_t> nextlevel;
    nextlevel.emplace(source_id);
    py::list res = py::list();
    while (nextlevel.size()) {
        std::unordered_set<node_t> thislevel = nextlevel;
        nextlevel = std::unordered_set<node_t>();
        for (std::unordered_set<node_t>::iterator i = thislevel.begin(); i != thislevel.end(); i++) {
            node_t v_id = *i;
            if (seen.find(v_id) == seen.end()) {
                seen.emplace(v_id);
                adj_attr_dict_factory& v_adj = G_adj[v_id];
                for (adj_attr_dict_factory::iterator j = v_adj.begin(); j != v_adj.end(); j++) {
                    node_t neighbor_id = j->first;
                    nextlevel.emplace(neighbor_id);
                }
            }
        }
    }

    for (std::unordered_set<node_t>::iterator i = seen.begin(); i != seen.end(); i++) {
        node_t res_id = *(i);
        res.append(G_.id_to_node.attr("get")(res_id));
    }
    return res;
}
