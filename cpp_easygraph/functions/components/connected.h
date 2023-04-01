#pragma once

#include "../../common/common.h"
#include "../../classes/linkgraph.h"

struct Edge_weighted{
    int toward, next;
};

py::object plain_bfs(py::object G, py::object source);
py::object connected_component_undirected(py::object G);
inline void _union_node(const int &u, const int &v, int *parent, int *rank_node);
int _getfa(const int & x, int *parent);
py::object connected_component_directed(py::object G);
void _tarjan(const int &u, int *Time, int *cnt, int *Tot, std::vector<LinkEdge>& E, std::vector<int>& head, int *dfn, int *low, int *st, int *color, bool *in_stack, Edge_weighted *E_res, int *head_res, int *edge_number_res);
void _add_edge_res(const int &u, const int &v, Edge_weighted *E_res, int *head_res, int *edge_number_res);
