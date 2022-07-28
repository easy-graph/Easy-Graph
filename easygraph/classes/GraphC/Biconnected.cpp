#include "Biconnected.h"
#include "Utils.h"

Graph::node_t index_edge(std::vector<std::pair<Graph::node_t, Graph::node_t>>& edges, const std::pair<Graph::node_t, Graph::node_t>& target) {
    for (Graph::node_t i = 0;i < edges.size();i++) {
        std::pair<Graph::node_t, Graph::node_t>edge = edges[i];
        if ((edge.first == target.first) && (edge.second == target.second)) {
            return i;
        }
    }
    return -1;
}

 
py::object _generator_biconnected_components_edges(py::object G, bool need_components = true) {
    py::list ret = py::list();
    std::set<Graph::node_t> visited;
    Graph& G_ = py::extract<Graph&>(G);
    Graph::node_dict_factory nodes_list = G_.node;
    for (Graph::node_dict_factory::iterator iter = nodes_list.begin();iter != nodes_list.end();iter++) {
        Graph::node_t start_id = iter->first;
        if (visited.find(start_id) != visited.end()) {
            continue;
        }
        std::unordered_map<Graph::node_t, Graph::node_t> discovery;
        std::unordered_map<Graph::node_t, Graph::node_t> low;
        Graph::node_t root_children = 0;
        discovery.emplace(start_id, 0);
        low.emplace(start_id, 0);
        visited.emplace(start_id);
        std::vector<std::pair<Graph::node_t, Graph::node_t>> edge_stack;
        std::vector<stack_node> stack;
        Graph::adj_attr_dict_factory start_adj = G_.adj[start_id];
        NeighborIterator neighbors_iter = NeighborIterator(start_adj);
        stack_node initial_stack_node(start_id, start_id, neighbors_iter);
        stack.emplace_back(initial_stack_node);
        while (!stack.empty()) {
            stack_node& node_info = stack.back();
            Graph::node_t node_grandparent_id = node_info.grandparent;
            Graph::node_t node_parent_id = node_info.parent;
            try {
                Graph::node_t node_child_id = node_info.neighbors_iter.next();
                if (node_grandparent_id == node_child_id) {
                    continue;
                }
                if (visited.find(node_child_id) != visited.end()) {
                    if (discovery[node_child_id] <= discovery[node_parent_id]) {
                        low[node_parent_id] = std::min(low[node_parent_id], discovery[node_child_id]);
                        if (need_components) {
                            edge_stack.emplace_back(std::make_pair(node_parent_id, node_child_id));
                        }
                    }
                }
                else {
                    low[node_child_id] = discovery[node_child_id] = discovery.size();
                    visited.emplace(node_child_id);
                    Graph::adj_attr_dict_factory node_child_adj = G_.adj[node_child_id];
                    NeighborIterator child_neighbors_iter = NeighborIterator(G_.adj[node_child_id]);
                    stack_node new_stack_node(node_parent_id, node_child_id, child_neighbors_iter);
                    stack.emplace_back(new_stack_node);
                    if (need_components) {
                        edge_stack.emplace_back(std::make_pair(node_parent_id, node_child_id));
                    }
                }
            }
            catch (int) {
                stack.pop_back();
                if (stack.size() > 1) {
                    if (low[node_parent_id] >= discovery[node_grandparent_id]) {
                        if (need_components) {
                            py::list tmp_ret = py::list();
                            std::pair<Graph::node_t, Graph::node_t> iter_edge = std::make_pair(-1, -1);
                            while ((iter_edge.first != node_grandparent_id || iter_edge.second != node_parent_id)) {
                                iter_edge = edge_stack.back();
                                edge_stack.pop_back();
                                tmp_ret.append(py::make_tuple(G_.id_to_node[iter_edge.first], G_.id_to_node[iter_edge.second]));
                            }
                            ret.append(tmp_ret);
                        }
                        else {
                            ret.append(G_.id_to_node[node_grandparent_id]);
                        }
                    }
                    low[node_grandparent_id] = std::min(low[node_grandparent_id], low[node_parent_id]);
                }
                else if (stack.size() > 0) {
                    root_children += 1;
                    if (need_components == true) {
                        std::pair<Graph::node_t, Graph::node_t> target = std::make_pair(node_grandparent_id, node_parent_id);
                        Graph::node_t ind = index_edge(edge_stack, target);
                        if (ind != -1) {
                            py::list tmp_ret = py::list();
                            for (Graph::node_t z = ind;z < edge_stack.size();z++) {
                                tmp_ret.append(py::make_tuple(G_.id_to_node[edge_stack[z].first], G_.id_to_node[edge_stack[z].second]));
                            }
                            ret.append(tmp_ret);
                        }
                    }
                }
            }
        }
        if (!need_components) {
            if (root_children > 1) {
                ret.append(G_.id_to_node(start_id));
            }
        }
    }
    return ret;
}

py::object generator_biconnected_components_edges(py::object G) {

    return _generator_biconnected_components_edges(G, true);

}
