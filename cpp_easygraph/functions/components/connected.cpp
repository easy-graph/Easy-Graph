#include "connected.h"

#include <pybind11/stl.h>

#include "../../classes/graph.h"
#include "../../classes/directed_graph.h"
#include "../../common/utils.h"
#include "time.h"

#define gmin(x, y) x = x < y? x: y

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

py::object connected_component_undirected(py::object G) {
    Graph& G_ = G.cast<Graph&>();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    if (is_directed == true) {
        printf("connected_component_undirected is designed for undirected graphs.\n");
        return py::dict();
    }
    int N = G_.node.size();
    int M = G.attr("number_of_edges")().cast<int>();
    std::vector<Edge_weighted> E_res(N+5);
    for(int i=0; i < N+5; i++) {
        E_res[i].toward = 0;
        E_res[i].next = 0;
    }
    int edge_number_res = 0;
    std::vector<int> parent(N+5, 0);
    std::vector<int> rank_node(N+5, 0);
    std::vector<int> color(N+5, 0);
    std::vector<int> head_res(N+5, 0);
    std::vector<bool> has_edge(N+5, false);


    py::list nodes_list = py::list(G.attr("nodes"));
    for (int i = 0;i < py::len(nodes_list);i++) {
        node_t i_id = (G_.node_to_id[nodes_list[i]]).cast<node_t>();
        parent[i_id] = i_id;
    }

    for (graph_edge& edge : G_._get_edges()) {
        node_t u = edge.u, v = edge.v;
        has_edge[u] = true; has_edge[v] = true;
        _union_node(u, v, parent, rank_node);
    }

    int Tot = 0;
	for(int i = 1; i < N + 1; ++i) {
        if (!has_edge[i])
            continue;
		int fx = _getfa(i, parent);
		if(fx == i){
			color[++Tot] = fx;
		}
	}

    for (int i = 1; i < N + 1; ++i) {
        int fx = _getfa(i, parent);
        _add_edge_res(fx, i, E_res, head_res, &edge_number_res);
    }

    py::dict ret = py::dict();
    for (int i = 1; i <= Tot; ++i) {
        py::list tmp = py::list();
        for(int p = head_res[color[i]]; p; p = E_res[p].next){
            tmp.append(py::cast(E_res[p].toward));
        }
        ret[py::cast(i)] = tmp;
    }
    return ret;
}

inline void _union_node(const int &u, const int &v, std::vector<int> &parent, std::vector<int> &rank_node) {
    int x = _getfa(u, parent), y = _getfa(v, parent);    //先找到两个根节点
    if (rank_node[x] <= rank_node[y])
        parent[x] = y;
    else
        parent[y] = x;
    if (rank_node[x] == rank_node[y] && x != y)
        rank_node[y]++;                   //如果深度相同且根节点不同，则新的根节点的深度+1
}

int _getfa(const int & x, std::vector<int> &parent) {
    int r,k,t;
    r=x;
    while(parent[r]!=r)
        r=parent[r];
    k=r;
    r=x;
    while(parent[r]!=k) {
        t=parent[r];
        parent[r]=k;
        r=t;
    }
    return k;
}

py::object connected_component_directed(py::object G) {
    bool is_directed = G.attr("is_directed")().cast<bool>();
    if (is_directed == false) {
        printf("connected_component_directed is designed for directed graphs.\n");
        return py::list();
    }
    DiGraph& G_ = G.cast<DiGraph&>();
    int N = G_.node.size();
    Graph_L G_l;
    if(G_.linkgraph_dirty || G_.linkgraph_structure.max_deg == -1){
        G_l = graph_to_linkgraph(G_, is_directed, "", true, false);
        G_.linkgraph_dirty = false;
    }
    else{
        G_l = G_.linkgraph_structure;
    }
    std::vector<LinkEdge>& E = G_l.edges;
    std::vector<int> outDegree = G_l.degree;
    std::vector<int> head = G_l.head;

    int Time = 0, cnt = 0, Tot = 0, edge_number_res = 0;

    std::vector<int> dfn(N+5, 0);
    std::vector<int> low(N+5, 0);
    std::vector<int> st(N+5, 0);
    std::vector<int> color(N+5, 0);
    std::vector<int> head_res(N+5, 0);
    std::vector<bool> in_stack(N+5, false);
    std::vector<bool> has_edge(N+5, false);

    std::vector<Edge_weighted> E_res(N+5);
    for(int i=0; i < N+5; i++) {
        E_res[i].toward = 0;
        E_res[i].next = 0;
    }

    for (graph_edge& edge : G_._get_edges()) {
        node_t u = edge.u, v = edge.v;
        has_edge[u] = true; has_edge[v] = true;
    }

    for (int i = 1; i < N + 1; ++i)
        if (!dfn[i] && has_edge[i])
            _tarjan(i, &Time, &cnt, &Tot, E, head, dfn, low, st, color, in_stack, E_res, head_res, &edge_number_res);

    py::list ret = py::list();
    for (int i = 1; i <= Tot; ++i) {
        py::set tmp;
        for(int p = head_res[i]; p; p = E_res[p].next)
            tmp.add(G_.id_to_node.attr("get")(E_res[p].toward));
        ret.append(tmp);
    }
    return ret;
}

void _add_edge_res(const int &u, const int &v, std::vector<Edge_weighted> &E_res, std::vector<int> &head_res, int *edge_number_res) {
    E_res[++(*edge_number_res)].next = head_res[u];
    E_res[*edge_number_res].toward = v;
    head_res[u] = *edge_number_res;
}

void _tarjan(const int &u, int *Time, int *cnt, int *Tot, std::vector<LinkEdge>& E, std::vector<int>& head, std::vector<int> &dfn, std::vector<int> &low, std::vector<int> &st, std::vector<int> &color, std::vector<bool> &in_stack, std::vector<Edge_weighted> &E_res, std::vector<int> &head_res, int *edge_number_res) {
    dfn[u] = low[u] = ++(*Time); st[++(*cnt)] = u; in_stack[u] = true;
    for(int p = head[u]; p != -1; p = E[p].next){
        int v = E[p].to;
        if (!dfn[v]) _tarjan(v, Time, cnt, Tot, E, head, dfn, low, st, color, in_stack, E_res, head_res, edge_number_res), gmin(low[u], low[v]); 
        else if (in_stack[v]) gmin(low[u], dfn[v]);
    }
    
    if (dfn[u] == low[u]) {
        for (++(*Tot); st[*cnt] != u; --(*cnt)) {
            _add_edge_res(*Tot, st[*cnt], E_res, head_res, edge_number_res);
            in_stack[st[*cnt]] = false, color[st[*cnt]] = *Tot;
        } 
        _add_edge_res(*Tot, st[*cnt], E_res, head_res, edge_number_res);
        in_stack[u] = false; color[u] = *Tot; --(*cnt);
    }
}
