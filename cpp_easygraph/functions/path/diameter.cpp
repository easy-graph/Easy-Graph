#include "path.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"

#include <vector>
#include <queue>
#include <limits>
#include <string>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ... (保留之前的 _bfs_eccentricity 函数不变) ...
double _bfs_eccentricity(const Graph_L& G_l, int source) {
    int N = G_l.n;
    std::vector<int> dis(N + 1, -1);
    std::queue<int> q;

    dis[source] = 0;
    q.push(source);

    double max_dist = 0.0;
    int visited_count = 0;

    const std::vector<int>& head = G_l.head;
    const std::vector<LinkEdge>& E = G_l.edges;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        visited_count++;
        
        if (dis[u] > max_dist) {
            max_dist = dis[u];
        }

        for (int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            if (dis[v] == -1) {
                dis[v] = dis[u] + 1;
                q.push(v);
            }
        }
    }
    
    return (visited_count == N) ? max_dist : -1.0;
}


py::object eccentricity(py::object G, py::object v = py::none(), py::object sp = py::none()) {
    Graph& G_ = G.cast<Graph&>();
    // 兼容判定有向图
    bool is_directed = py::hasattr(G, "is_directed") ? G.attr("is_directed")().cast<bool>() : false;
    
    // 💡 修复点 1：不使用 .order()，改用 len(G.nodes)，这在 Graph 和 GraphC 中绝对通用
    int order = py::len(G.attr("nodes"));

    if (!sp.is_none()) {
        py::dict result;
        py::dict sp_dict = sp.cast<py::dict>();
        
        py::list nodes_to_check;
        bool is_single = false;
        
        // 兼容判断 v 参数类型
        if (v.is_none()) {
            nodes_to_check = py::list(G.attr("nodes"));
        } else if (py::isinstance<py::list>(v) || py::isinstance<py::tuple>(v) || py::isinstance<py::set>(v)) {
            nodes_to_check = py::list(v);
        } else {
            nodes_to_check.append(v);
            is_single = true;
        }

        for (py::handle n : nodes_to_check) {
            if (!sp_dict.contains(n)) {
                throw py::type_error("Format of 'sp' is invalid.");
            }
            py::dict length_dict = sp_dict[n].cast<py::dict>();
            if (py::len(length_dict) != order) {
                std::string msg = is_directed ? 
                    "Found infinite path length because the digraph is not strongly connected" :
                    "Found infinite path length because the graph is not connected";
                throw py::value_error(msg);
            }
            
            double max_val = 0.0;
            for (auto item : length_dict) {
                double val = item.second.cast<double>();
                if (val > max_val) max_val = val;
            }
            result[n] = max_val;
        }
        
        if (is_single) return result[v];
        return result;
    }

    // ========== sp=None (OpenMP 加速) ==========

    if (G_.linkgraph_dirty) {
        G_.linkgraph_structure = graph_to_linkgraph(G_, is_directed, "", true, false);
        G_.linkgraph_dirty = false;
    }
    const Graph_L& G_l = G_.linkgraph_structure;

    int N = G_l.n;
    if (N == 0) return py::dict();

    // 💡 修复点 2：手动构建 ID ↔ Node 映射，不再依赖不稳定的 .node_index 和 .index_node
    py::dict node_index;
    std::vector<py::object> index_to_node_vec;

    if (py::hasattr(G, "node_index")) {
        node_index = G.attr("node_index").cast<py::dict>();
    } else {
        // Fallback: 如果是 GraphC 没有 node_index，我们自己按遍历顺序生成一个
        int idx = 0;
        for (py::handle n : G.attr("nodes")) {
            node_index[n] = py::cast(idx++);
        }
    }

    if (py::hasattr(G, "index_node")) {
        py::object idx_n = G.attr("index_node");
        if (py::isinstance<py::list>(idx_n)) {
            py::list l = idx_n.cast<py::list>();
            for (auto item : l) index_to_node_vec.push_back(item.cast<py::object>());
        } else if (py::isinstance<py::dict>(idx_n)) {
            py::dict d = idx_n.cast<py::dict>();
            index_to_node_vec.resize(py::len(d));
            for (auto item : d) {
                index_to_node_vec[item.first.cast<int>()] = item.second.cast<py::object>();
            }
        }
    } else {
        // Fallback: 如果是 GraphC 没有 index_node，我们通过刚才的 node_index 反推
        index_to_node_vec.resize(py::len(node_index));
        for (auto item : node_index) {
            index_to_node_vec[item.second.cast<int>()] = item.first.cast<py::object>();
        }
    }

    // 解析 v 参数为 C++ ID 列表
    std::vector<int> target_ids;
    bool is_single_node = false;

    if (v.is_none()) {
        target_ids.resize(N);
        for (int i = 0; i < N; ++i) target_ids[i] = i + 1;
    } else if (py::isinstance<py::list>(v) || py::isinstance<py::tuple>(v) || py::isinstance<py::set>(v)) {
        py::iterable v_iterable = v.cast<py::iterable>();
        for (py::handle node : v_iterable) {
            if (node_index.contains(node)) {
                target_ids.push_back(node_index[node].cast<int>() + 1);
            }
        }
    } else {
        if (node_index.contains(v)) {
            is_single_node = true;
            target_ids.push_back(node_index[v].cast<int>() + 1);
        } else {
            throw py::value_error("Node not found in graph.");
        }
    }

    int num_targets = target_ids.size();
    if (num_targets == 0) return py::dict();

    std::vector<double> ecc_values(num_targets, -1.0);
    bool is_connected = true;

    {
        py::gil_scoped_release release;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_targets; i++) {
            if (!is_connected) continue;
            int u = target_ids[i];
            
            double ecc_val = _bfs_eccentricity(G_l, u); 

            if (ecc_val < 0) {
                #pragma omp critical
                is_connected = false;
            } else {
                ecc_values[i] = ecc_val;
            }
        }
    }

    if (!is_connected) {  
        std::string msg = is_directed ? 
            "Found infinite path length because the digraph is not strongly connected" :
            "Found infinite path length because the graph is not connected";
        throw py::value_error(msg);
    }

    // 打包返回
    if (is_single_node) {
        return py::cast(ecc_values[0]);
    } else {
        py::dict result;
        for (int i = 0; i < num_targets; i++) {
            int internal_id = target_ids[i];
            // ID 还原为节点名称 (注意 internal_id 是 1-based, std::vector 是 0-based)
            py::object original_node = index_to_node_vec[internal_id - 1]; 
            result[original_node] = ecc_values[i];
        }
        return result;
    }
}