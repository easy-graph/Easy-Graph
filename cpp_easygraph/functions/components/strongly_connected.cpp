#include "strongly_connected.h"
#include "connected.h"

#include "../../classes/directed_graph.h"

#define MAX_NODES_NUM4RECURSION_METHOD 100000

py::object strongly_connected_components(py::object G) {
    bool is_directed = G.attr("is_directed")().cast<bool>();
    if (is_directed == false) {
        printf("connected_component_directed is designed for directed graphs.\n");
        return py::list();
    }
    
    int N = G.attr("number_of_nodes")().cast<int>();
    if(N < MAX_NODES_NUM4RECURSION_METHOD){
        return connected_component_directed(G);
    }

    return strongly_connected_components_iteration_impl(G);
}

py::object strongly_connected_components_iteration_impl(py::object G) {
    py::list res = py::list();
    DiGraph& G_ = py::cast<DiGraph&>(G);
    adj_dict_factory& adj = G_.adj;
    std::unordered_map<node_t, node_t> preorder;
    std::unordered_map<node_t, node_t> lowlink;
    std::set<node_t> scc_found;
    std::vector<node_t> scc_queue;
    int i = 0;
    node_dict_factory& nodes_list = G_.node;
    for (node_dict_factory::iterator source = nodes_list.begin(); source != nodes_list.end(); source++) {
        node_t source_id = source->first;
        if (scc_found.find(source_id) == scc_found.end()) {
            std::vector<node_t> que;
            que.emplace_back(source_id);
            while (!que.empty()) {
                node_t v_id = que.back();
                if (preorder.find(v_id) == preorder.end()) {
                    i += 1;
                    preorder[v_id] = i;
                }
                bool done = true;
                adj_attr_dict_factory& v_neighbors = adj[v_id];
                for (adj_attr_dict_factory::iterator w = v_neighbors.begin(); w != v_neighbors.end(); w++) {
                    node_t w_id = w->first;
                    if (preorder.find(w_id) == preorder.end()) {
                        que.emplace_back(w_id);
                        done = false;
                        break;
                    }
                }
                if (done) {
                    lowlink[v_id] = preorder[v_id];
                    for (adj_attr_dict_factory::iterator w = v_neighbors.begin(); w != v_neighbors.end(); w++) {
                        node_t w_id = w->first;
                        if (scc_found.find(w_id) == scc_found.end()) {
                            if (preorder[w_id] > preorder[v_id]) {
                                lowlink[v_id] = std::min(lowlink[v_id], lowlink[w_id]);
                            } else {
                                lowlink[v_id] = std::min(lowlink[v_id], preorder[w_id]);
                            }
                        }
                    }
                    que.pop_back();
                    if (lowlink[v_id] == preorder[v_id]) {
                        std::unordered_set<node_t> scc;
                        scc.emplace(v_id);
                        while (!scc_queue.empty() && (preorder[scc_queue.back()] > preorder[v_id])) {
                            node_t k = scc_queue.back();
                            scc_queue.pop_back();
                            scc.emplace(k);
                        }
                        
                        py::set tmp_res;
                        for (std::unordered_set<node_t>::iterator z = scc.begin(); z != scc.end(); z++) {
                            scc_found.emplace(*z);
                            tmp_res.add(G_.id_to_node.attr("get")(*z));
                        }
                        res.append(tmp_res);
                    } else {
                        scc_queue.emplace_back(v_id);
                    }
                }
            }
        }
    }
    return res;
}
