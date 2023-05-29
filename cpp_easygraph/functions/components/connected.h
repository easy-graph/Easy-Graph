#pragma once

#include "../../common/common.h"
#include "../../classes/linkgraph.h"

struct Edge_weighted{
    int toward, next;
};

py::object plain_bfs(py::object G, py::object source);
py::object connected_component_undirected(py::object G);
inline void _union_node(const int &u, const int &v, std::vector<int>& parent, std::vector<int> &rank_node);
int _getfa(const int & x, std::vector<int> &parent);
py::object connected_component_directed(py::object G);
void _tarjan(const int &u, int *Time, int *cnt, int *Tot, std::vector<LinkEdge>& E, std::vector<int>& head, std::vector<int> &dfn, std::vector<int> &low, std::vector<int> &st, std::vector<int> &color, std::vector<bool> &in_stack, std::vector<Edge_weighted> &E_res, std::vector<int> &head_res, int *edge_number_res);
void _add_edge_res(const int &u, const int &v, std::vector<Edge_weighted> &E_res, std::vector<int> &head_res, int *edge_number_res);
