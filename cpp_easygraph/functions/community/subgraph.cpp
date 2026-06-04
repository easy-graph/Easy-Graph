#include "subgraph.h"
#include "../../classes/graph.h"
#include "../../classes/directed_graph.h"
#include "../../common/utils.h"

static node_t _add_one_node_for_subgraph(Graph& self, py::object one_node_for_adding, py::object node_attr = py::dict()) {
    node_t id;
    if (self.node_to_id.contains(one_node_for_adding)) {
        id = self.node_to_id[one_node_for_adding].cast<node_t>();
    } else {
        id = ++(self.id);
        self.id_to_node[py::cast(id)] = one_node_for_adding;
        self.node_to_id[one_node_for_adding] = id;
    }
    py::list items = py::list(node_attr.attr("items")());
    self.node[id] = node_attr_dict_factory();
    for (int i = 0; i < len(items); i++) {
        py::tuple kv = items[i].cast<py::tuple>();
        py::object pkey = kv[0];
        std::string weight_key = weight_to_string(pkey);
        weight_t value = kv[1].cast<weight_t>();
        self.node[id].insert(std::make_pair(weight_key, value));
    }
    return id;
}

static void _add_one_edge_for_subgraph(Graph& self, node_t u, node_t v, py::object edge_attr) {
    py::list items = py::list(edge_attr.attr("items")());
    self.adj[u][v] = node_attr_dict_factory();
    self.adj[v][u] = node_attr_dict_factory();
    for (int i = 0; i < len(items); i++) {
        py::tuple kv = items[i].cast<py::tuple>();
        py::object pkey = kv[0];
        std::string weight_key = weight_to_string(pkey);
        weight_t value = kv[1].cast<weight_t>();
        self.adj[u][v].insert(std::make_pair(weight_key, value));
        self.adj[v][u].insert(std::make_pair(weight_key, value));
    }
}

static py::object _nodes_subgraph_cpp_graph(py::object self, std::vector<node_t>& node_ids) {
    Graph& self_ = self.cast<Graph&>();
    py::object G = self.attr("__class__")();
    Graph& G_ = G.cast<Graph&>();
    G_.graph.attr("update")(self_.graph);
    
    py::object nodes = self.attr("nodes");
    py::object adj = self.attr("adj");
    
    // 修复：维护原图内部索引到新图内部索引的映射
    std::unordered_map<node_t, node_t> old_id_to_new_id;
    
    // 第一遍：添加所有节点并建立映射
    for (node_t node_id : node_ids) {
        py::object node = self_.id_to_node[py::cast(node_id)];
        py::object node_attr = nodes[node];
        node_t new_id = _add_one_node_for_subgraph(G_, node, node_attr);
        old_id_to_new_id[node_id] = new_id;
    }
    
    // 第二遍：添加边（使用映射后的索引）
    for (node_t node_id : node_ids) {
        py::object node = self_.id_to_node[py::cast(node_id)];
        py::object out_edges = adj[node];
        py::list edge_items = py::list(out_edges.attr("items")());
        
        // 获取当前节点的新索引
        node_t new_u = old_id_to_new_id[node_id];
        
        for (int j = 0; j < py::len(edge_items); j++) {
            py::tuple item = edge_items[j].cast<py::tuple>();
            py::object v = item[0];
            py::object edge_attr = item[1];
            
            if (self_.node_to_id.contains(v)) {
                node_t v_id = self_.node_to_id[v].cast<node_t>();
                bool v_in_subgraph = std::find(node_ids.begin(), node_ids.end(), v_id) != node_ids.end();
                if (v_in_subgraph) {
                    // 使用映射后的索引
                    node_t new_v = old_id_to_new_id[v_id];
                    _add_one_edge_for_subgraph(G_, new_u, new_v, edge_attr);
                }
            }
        }
    }
    
    return G;
}

static node_t DiGraph_add_one_node_for_subgraph(DiGraph& self, py::object one_node_for_adding, py::object node_attr = py::dict()) {
    node_t id;
    if (self.node_to_id.contains(one_node_for_adding)) {
        id = self.node_to_id[one_node_for_adding].cast<node_t>();
    } else {
        id = ++(self.id);
        self.id_to_node[py::cast(id)] = one_node_for_adding;
        self.node_to_id[one_node_for_adding] = id;
    }
    py::list items = py::list(node_attr.attr("items")());
    self.node[id] = node_attr_dict_factory();
    for (int i = 0; i < len(items); i++) {
        py::tuple kv = items[i].cast<py::tuple>();
        py::object pkey = kv[0];
        std::string weight_key = weight_to_string(pkey);
        weight_t value = kv[1].cast<weight_t>();
        self.node[id].insert(std::make_pair(weight_key, value));
    }
    return id;
}

static void DiGraph_add_one_edge_for_subgraph(DiGraph& self, node_t u, node_t v, py::object edge_attr) {
    py::list items = py::list(edge_attr.attr("items")());
    self.adj[u][v] = node_attr_dict_factory();
    for (int i = 0; i < len(items); i++) {
        py::tuple kv = items[i].cast<py::tuple>();
        py::object pkey = kv[0];
        std::string weight_key = weight_to_string(pkey);
        weight_t value = kv[1].cast<weight_t>();
        self.adj[u][v].insert(std::make_pair(weight_key, value));
    }
}

static py::object _nodes_subgraph_cpp_digraph(py::object self, std::vector<node_t>& node_ids) {
    DiGraph& self_ = self.cast<DiGraph&>();
    py::object G = self.attr("__class__")();
    DiGraph& G_ = G.cast<DiGraph&>();
    G_.graph.attr("update")(self_.graph);
    
    py::object nodes = self.attr("nodes");
    py::object adj = self.attr("adj");
    
    // 修复：维护原图内部索引到新图内部索引的映射
    std::unordered_map<node_t, node_t> old_id_to_new_id;
    
    // 第一遍：添加所有节点并建立映射
    for (node_t node_id : node_ids) {
        py::object node = self_.id_to_node[py::cast(node_id)];
        py::object node_attr = nodes[node];
        node_t new_id = DiGraph_add_one_node_for_subgraph(G_, node, node_attr);
        old_id_to_new_id[node_id] = new_id;
    }
    
    // 第二遍：添加边（使用映射后的索引）
    for (node_t node_id : node_ids) {
        py::object node = self_.id_to_node[py::cast(node_id)];
        py::object out_edges = adj[node];
        py::list edge_items = py::list(out_edges.attr("items")());
        
        // 获取当前节点的新索引
        node_t new_u = old_id_to_new_id[node_id];
        
        for (int j = 0; j < py::len(edge_items); j++) {
            py::tuple item = edge_items[j].cast<py::tuple>();
            py::object v = item[0];
            py::object edge_attr = item[1];
            
            if (self_.node_to_id.contains(v)) {
                node_t v_id = self_.node_to_id[v].cast<node_t>();
                bool v_in_subgraph = std::find(node_ids.begin(), node_ids.end(), v_id) != node_ids.end();
                if (v_in_subgraph) {
                    // 使用映射后的索引
                    node_t new_v = old_id_to_new_id[v_id];
                    DiGraph_add_one_edge_for_subgraph(G_, new_u, new_v, edge_attr);
                }
            }
        }
    }
    
    return G;
}

py::object nodes_subgraph_cpp(py::object self, std::vector<node_t>& node_ids) {
    bool is_directed = self.attr("is_directed")().cast<bool>();
    if (is_directed) {
        return _nodes_subgraph_cpp_digraph(self, node_ids);
    } else {
        return _nodes_subgraph_cpp_graph(self, node_ids);
    }
}
