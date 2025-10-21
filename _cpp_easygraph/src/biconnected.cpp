#include "biconnected.h"
#include "graph.h"
#include "utils.h"

node_t index_edge(std::vector<std::pair<node_t, node_t>>& edges, const std::pair<node_t, node_t>& target) {
    for (int i = edges.size() - 1;i >= 0;i--) {
        if ((edges[i].first == target.first) && (edges[i].second == target.second)) {
            return i;
        }
    }
    return -1;
}


py::list _biconnected_dfs_record_edges(Graph& G, bool need_components) {
    py::list ret;
    std::unordered_set<node_t> visited;

    auto& nodes = G.node;   // id -> node attributes
    auto& adj   = G.adj;    // id -> (neighbor id -> edge attrs)

    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        const node_t start_id = it->first;
        if (visited.find(start_id) != visited.end()) {
            continue;
        }

        std::unordered_map<node_t, int> discovery;
        std::unordered_map<node_t, int> low;
        node_t root_children = 0;

        discovery.emplace(start_id, 0);
        low.emplace(start_id, 0);
        visited.emplace(start_id);

        std::vector<std::pair<node_t, node_t>> edge_stack;
        std::vector<stack_node> stack;

        // 注意：这里与原实现一致，使用 operator[] 获取邻接（如无则创建空条目）
        adj_attr_dict_factory start_adj = adj[start_id];
        NeighborIterator neighbors_iter(start_adj);
        stack.emplace_back(stack_node{start_id, start_id, neighbors_iter});

        while (!stack.empty()) {
            stack_node& node_info = stack.back();
            const node_t node_grandparent_id = node_info.grandparent;
            const node_t node_parent_id      = node_info.parent;

            try {
                const node_t node_child_id = node_info.neighbors_iter.next();

                if (node_grandparent_id == node_child_id) {
                    continue;
                }

                if (visited.find(node_child_id) != visited.end()) {
                    // 回边：更新 low值
                    if (discovery[node_child_id] <= discovery[node_parent_id]) {
                        low[node_parent_id] = std::min(low[node_parent_id], discovery[node_child_id]);
                        if (need_components) {
                            edge_stack.emplace_back(node_parent_id, node_child_id);
                        }
                    }
                } else {
                    // 树边：首次发现
                    low[node_child_id] = discovery[node_child_id] = static_cast<int>(discovery.size());
                    visited.emplace(node_child_id);

                    // 子节点邻居迭代器
                    NeighborIterator child_neighbors_iter(adj[node_child_id]);

                    // 压栈：当前 parent 变为 grandparent，新 child 为 parent
                    stack.emplace_back(stack_node{node_parent_id, node_child_id, child_neighbors_iter});

                    if (need_components) {
                        edge_stack.emplace_back(node_parent_id, node_child_id);
                    }
                }
            } catch (int) {
                // 当前迭代器穷尽：弹栈并做割点/双连通分量处理
                stack.pop_back();

                if (stack.size() > 1) {
                    // 非根结点情况：检查割点条件
                    if (low[node_parent_id] >= discovery[node_grandparent_id]) {
                        if (need_components) {
                            py::list tmp_ret;
                            std::pair<node_t, node_t> iter_edge(-1, -1);
                            while (iter_edge.first != node_grandparent_id || iter_edge.second != node_parent_id) {
                                iter_edge = edge_stack.back();
                                edge_stack.pop_back();
                                // 通过 id_to_node 还原 Python 节点对象
                                py::object u = G.id_to_node.attr("__getitem__")(py::cast(iter_edge.first));
                                py::object v = G.id_to_node.attr("__getitem__")(py::cast(iter_edge.second));
                                tmp_ret.append(py::make_tuple(u, v));
                            }
                            ret.append(tmp_ret);
                        } else {
                            py::object ap = G.id_to_node.attr("__getitem__")(py::cast(node_grandparent_id));
                            ret.append(ap);  // 记录割点
                        }
                    }
                    // 回溯时更新祖父节点的 low
                    low[node_grandparent_id] = std::min(low[node_grandparent_id], low[node_parent_id]);
                } else if (!stack.empty()) {
                    // 根的一个子树处理完毕
                    ++root_children;
                    if (need_components) {
                        const std::pair<node_t, node_t> target(node_grandparent_id, node_parent_id);
                        const node_t ind = index_edge(edge_stack, target);
                        if (ind != static_cast<node_t>(-1)) {
                            py::list tmp_ret;
                            for (node_t z = ind; z < edge_stack.size(); ++z) {
                                const auto& e = edge_stack[z];
                                py::object u = G.id_to_node.attr("__getitem__")(py::cast(e.first));
                                py::object v = G.id_to_node.attr("__getitem__")(py::cast(e.second));
                                tmp_ret.append(py::make_tuple(u, v));
                            }
                            ret.append(tmp_ret);
                        }
                    }
                }
            }
        } // while stack

        if (!need_components) {
            if (root_children > 1) {
                py::object root = G.id_to_node.attr("__getitem__")(py::cast(start_id));
                ret.append(root);  // 根是割点
            }
        }
    } // for nodes

    return ret;
}
