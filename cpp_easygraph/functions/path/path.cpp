#include "path.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef EASYGRAPH_ENABLE_GPU
#include <gpu_easygraph.h>
#endif

#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include "../../classes/segment_tree.cpp"


#include <vector>
#include <queue>
#include <limits>
#include <string>


std::vector<double> _dijkstra(const Graph_L& G_l, int source, int target) {
    int N = G_l.n;
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dis(N + 1, INF);
    std::priority_queue<std::pair<double, int>, 
                        std::vector<std::pair<double, int>>, 
                        std::greater<std::pair<double, int>>> pq;

    dis[source] = 0.0;
    pq.push({0.0, source});

    const std::vector<int>& head = G_l.head;
    const std::vector<LinkEdge>& E = G_l.edges;

    while (!pq.empty()) {
        std::pair<double, int> top = pq.top();
        pq.pop();
        
        double d = top.first;
        int u = top.second;

        //Lazy deletion
        if (d > dis[u]) continue;

        // cutoff
        if (u == target) break;

        for (int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            double w = static_cast<double>(E[p].w); 
            
            if (dis[u] + w < dis[v]) {
                dis[v] = dis[u] + w;
                pq.push({dis[v], v});
            }
        }
    }
    return dis;
}


py::object _invoke_cpp_dijkstra_multisource(py::object G, py::object sources, py::object weight, py::object target) {
    bool is_directed = G.attr("is_directed")().cast<bool>();
    Graph& G_ = G.cast<Graph&>();
    std::string weight_key = weight_to_string(weight);
    
    Graph_L G_l;
    if(G_.linkgraph_dirty){
        G_l = graph_to_linkgraph(G_, is_directed, weight_key, true, false);
        G_.linkgraph_structure = G_l;
        G_.linkgraph_dirty = false;
    } else {
        G_l = G_.linkgraph_structure;
    }

    node_t target_id = -1;
    if (!target.is_none()) {
        target_id = G_.node_to_id.attr("get")(target, -1).cast<node_t>();
    }

    py::list sources_list = py::list(sources);
    int num_sources = py::len(sources_list);
    int N = G_l.n;
    
    std::vector<node_t> source_ids(num_sources);
    for(int i = 0; i < num_sources; i++){
        if(G_.node_to_id.attr("get")(sources_list[i], py::none()).is(py::none())){
            printf("The node should exist in the graph!");
            return py::none();
        }
        source_ids[i] = G_.node_to_id.attr("get")(sources_list[i]).cast<node_t>();
    }
    std::vector<double> results(num_sources * N);
    {
        py::gil_scoped_release release;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_sources; i++) {
            node_t s = source_ids[i];

            std::vector<double> dists = _dijkstra(G_l, s, target_id);
            
            size_t offset = (size_t)i * N; 
            for (int j = 1; j <= N; j++) {
                results[offset + (j - 1)] = dists[j];
            }
        }
    } 

    py::array::ShapeContainer ret_shape{num_sources, N};
    py::array_t<double> ret(ret_shape, results.data());

    return ret;
}

#ifdef EASYGRAPH_ENABLE_GPU
py::object _invoke_gpu_dijkstra_multisource(py::object G,py::object py_sources, py::object weight, py::object target) {
    Graph& G_ = G.cast<Graph&>();
    if (weight.is_none()) {
        G_.gen_CSR();
    } else {
        G_.gen_CSR(weight_to_string(weight));
    }
    auto csr_graph = G_.csr_graph;
    std::vector<int>& E = csr_graph->E;
    std::vector<int>& V = csr_graph->V;
    std::vector<double> *W_p = weight.is_none() ? &(csr_graph->unweighted_W)
                                : csr_graph->W_map.find(weight_to_string(weight))->second.get();
    auto sources = G_.gen_CSR_sources(py_sources);
    std::vector<double> sssp;
    int gpu_r = gpu_easygraph::sssp_dijkstra(V, E, *W_p, *sources,
                                            target.is_none() ? -1 : (int)py::cast<py::int_>(target),
                                            sssp);

    if (gpu_r != gpu_easygraph::EG_GPU_SUCC) {
        // the code below will throw an exception
        py::pybind11_fail(gpu_easygraph::err_code_detail(gpu_r));
    }

    py::array::ShapeContainer ret_shape{(int)sources->size(), (int)V.size() - 1};
    py::array_t<double> ret(ret_shape, sssp.data());

    return ret;
}
#endif

py::object _dijkstra_multisource(py::object G,py::object sources, py::object weight, py::object target) {
#ifdef EASYGRAPH_ENABLE_GPU
    return _invoke_gpu_dijkstra_multisource(G, sources, weight, target);
#else
    return _invoke_cpp_dijkstra_multisource(G, sources, weight, target);
#endif
}


py::object _spfa(py::object G, py::object source, py::object weight) {
    Graph& G_ = G.cast<Graph&>();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    std::string weight_key = weight_to_string(weight);
    Graph_L G_l = graph_to_linkgraph(G_, is_directed,weight_key, false);
    int N = G_.node.size();

    std::vector<int> Q(N+10,0);
    std::vector<double> dis(N+1,INFINITY);
    std::vector<bool> vis(N+1,false);

    int l = 0, r = 1;
    node_t S = G_.node_to_id[source].cast<node_t>();
	Q[0] = S; vis[S] = true; dis[S] = 0;
    std::vector<LinkEdge>& E = G_l.edges;

    std::vector<int>& head = G_l.head;
    while (l != r) {
    	if (r != 0 && dis[Q[l]] >= dis[Q[r - 1]])
    		std::swap(Q[l], Q[r - 1]);
        int u = Q[l++];
        if (l >= N) l -= N;
        vis[u] = true;

        for(int p = head[u]; p != -1; p = E[p].next) {
            int v=E[p].to;
            if (dis[v]>dis[u]+E[p].w) {
                dis[v]=dis[u]+E[p].w;
                if (!vis[v]) {
                    vis[v]=true;
                    if (l == 0 || dis[v] >= dis[Q[l]])
						Q[r++]=v;
					else
						Q[--l]=v;
                    if (r >= N) r -= N;
                }
            }
        }
	}
    py::list pydist = py::list();
    for(int i = 1; i <= N; i++){
        pydist.append(py::cast(dis[i]));
    }
    return pydist;
}


py::object Prim(py::object G, py::object weight) {
    std::unordered_map<node_t, std::unordered_map<node_t, weight_t>> res_dict;
    py::dict result_dict = py::dict();
    Graph& G_ = G.cast<Graph&>();
    adj_dict_factory adj = G_.adj;
    std::vector<node_t> selected;
    std::vector<node_t> candidate;
    node_dict_factory& node_list = G_.node;
    std::string weight_key = weight_to_string(weight);
    for (node_dict_factory::iterator i = node_list.begin(); i != node_list.end(); i++) {
        node_t node_id = i->first;
        result_dict[G_.id_to_node[py::cast(node_id)]] = py::dict();
        if (selected.size() == 0) {
            selected.emplace_back(node_id);
        } else {
            candidate.emplace_back(node_id);
        }
    }
    while (candidate.size() > 0) {
        node_t start_id = -1;
        node_t end_id = -1;
        weight_t min_weight = INFINITY;
        int selected_len = selected.size();
        int candidate_len = candidate.size();
        for (int i = 0; i < selected_len; i++) {
            for (int j = 0; j < candidate_len; j++) {
                adj_attr_dict_factory node_adj = G_.adj[selected[i]];
                edge_attr_dict_factory edge_attr;
                weight_t edge_weight = INFINITY;
                bool j_exist = false;
                if (node_adj.find(candidate[j]) != node_adj.end()) {
                    edge_attr = node_adj[candidate[j]];
                    edge_weight = edge_attr.find(weight_key) != edge_attr.end() ? edge_attr[weight_key] : 1;
                    j_exist = true;
                }
                if ((node_list.find(selected[i]) != node_list.end()) &&
                    j_exist &&
                    (edge_weight < min_weight)) {
                    start_id = selected[i];
                    end_id = candidate[j];
                    min_weight = edge_weight;
                }
            }
        }
        if (start_id != -1 && end_id != -1) {
            res_dict[start_id][end_id] = min_weight;
            selected.emplace_back(end_id);
            std::vector<node_t>::iterator temp_iter;
            temp_iter = std::find(candidate.begin(), candidate.end(), end_id);
            candidate.erase(temp_iter);
        } else {
            break;
        }
    }
    for (std::unordered_map<node_t, std::unordered_map<node_t, weight_t>>::iterator k = res_dict.begin();
         k != res_dict.end(); k++) {
        py::object res_node = G_.id_to_node[py::cast(k->first)];
        for (std::unordered_map<node_t, weight_t>::iterator z = k->second.begin(); z != k->second.end(); z++) {
            py::object res_adj_node = G_.id_to_node[py::cast(z->first)];
            result_dict[res_node][res_adj_node] = z->second;
        }
    }
    return result_dict;
}
bool comp(const std::pair<std::pair<node_t, node_t>, weight_t>& a, const std::pair<std::pair<node_t, node_t>, weight_t>& b) {
    return a.second < b.second;
}
py::object Kruskal(py::object G, py::object weight) {
    std::unordered_map<node_t, std::unordered_map<node_t, weight_t>> res_dict;
    py::dict result_dict = py::dict();
    std::vector<std::vector<node_t>> group;
    Graph& G_ = G.cast<Graph&>();
    adj_dict_factory& adj = G_.adj;
    node_dict_factory& node_list = G_.node;
    std::vector<std::pair<std::pair<node_t, node_t>, weight_t>> edge_list;
    std::string weight_key = weight_to_string(weight);
    for (node_dict_factory::iterator i = node_list.begin(); i != node_list.end(); i++) {
        node_t i_id = i->first;
        result_dict[G_.id_to_node[py::cast(i_id)]] = py::dict();
        std::vector<node_t> temp_vector;
        temp_vector.emplace_back(i_id);
        group.emplace_back(temp_vector);
        adj_attr_dict_factory i_adj = adj[i_id];
        for (adj_attr_dict_factory::iterator j = i_adj.begin(); j != i_adj.end(); j++) {
            node_t j_id = j->first;
            weight_t edge_weight = adj[i_id][j_id].find(weight_key) != adj[i_id][j_id].end() ? adj[i_id][j_id][weight_key] : 1;
            edge_list.emplace_back(std::make_pair(std::make_pair(i_id, j_id), edge_weight));
        }
    }
    std::sort(edge_list.begin(), edge_list.end(), comp);
    node_t m, n;
    int group_size = group.size();
    for (auto edge : edge_list) {
        for (int i = 0; i < group_size; i++) {
            int group_i_size = group[i].size();
            for (int j = 0; j < group_i_size; j++) {
                if (group[i][j] == edge.first.first) {
                    m = i;
                    break;
                }
            }
            for (int j = 0; j < group_i_size; j++) {
                if (group[i][j] == edge.first.second) {
                    n = i;
                    break;
                }
            }
        }
        if (m != n) {
            res_dict[edge.first.first][edge.first.second] = edge.second;
            std::vector<node_t> temp_vector;
            group[m].insert(group[m].end(), group[n].begin(), group[n].end());
            group[n].clear();
        }
    }
    for (std::unordered_map<node_t, std::unordered_map<node_t, weight_t>>::iterator k = res_dict.begin();
         k != res_dict.end(); k++) {
        py::object res_node = G_.id_to_node[py::cast(k->first)];
        for (std::unordered_map<node_t, weight_t>::iterator z = k->second.begin(); z != k->second.end(); z++) {
            py::object res_adj_node = G_.id_to_node[py::cast(z->first)];
            result_dict[res_node][res_adj_node] = z->second;
        }
    }
    return result_dict;
}

py::object Floyd(py::object G, py::object weight) {
    std::unordered_map<node_t, std::unordered_map<node_t, weight_t>> res_dict;
    Graph& G_ = G.cast<Graph&>();
    adj_dict_factory& adj = G_.adj;
    py::dict result_dict = py::dict();
    node_dict_factory& node_list = G_.node;
    std::string weight_key = weight_to_string(weight);
    for (node_dict_factory::iterator i = node_list.begin(); i != node_list.end(); i++) {
        result_dict[G_.id_to_node[py::cast(i->first)]] = py::dict();
        adj_attr_dict_factory temp_key = adj[i->first];
        for (node_dict_factory::iterator j = node_list.begin(); j != node_list.end(); j++) {
            if (temp_key.find(j->first) != temp_key.end()) {
                if (adj[i->first][j->first].count(weight_key) == 0) {
                    adj[i->first][j->first][weight_key] = 1;
                }
                res_dict[i->first][j->first] = adj[i->first][j->first][weight_key];
            } else {
                res_dict[i->first][j->first] = INFINITY;
            }
            if (i->first == j->first) {
                res_dict[i->first][i->first] = 0;
            }
        }
    }

    for (node_dict_factory::iterator k = node_list.begin(); k != node_list.end(); k++) {
        for (node_dict_factory::iterator i = node_list.begin(); i != node_list.end(); i++) {
            for (node_dict_factory::iterator j = node_list.begin(); j != node_list.end(); j++) {
                weight_t temp = res_dict[i->first][k->first] + res_dict[k->first][j->first];
                weight_t i_j_weight = res_dict[i->first][j->first];
                if (i_j_weight > temp) {
                    res_dict[i->first][j->first] = temp;
                }
            }
        }
    }

    for (std::unordered_map<node_t, std::unordered_map<node_t, weight_t>>::iterator k = res_dict.begin();
         k != res_dict.end(); k++) {
        py::object res_node = G_.id_to_node[py::cast(k->first)];
        for (std::unordered_map<node_t, weight_t>::iterator z = k->second.begin(); z != k->second.end(); z++) {
            py::object res_adj_node = G_.id_to_node[py::cast(z->first)];
            result_dict[res_node][res_adj_node] = z->second;
        }
    }
    return result_dict;
}
