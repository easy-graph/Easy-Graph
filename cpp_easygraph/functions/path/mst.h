#pragma once

#include "../../common/common.h"

class UnionFind {
   public:
    UnionFind();
    UnionFind(std::vector<node_t> elements);
    node_t operator[](node_t object);
    void _union(node_t object1, node_t object2);

   private:
    std::unordered_map<node_t, node_t> parents;
    std::unordered_map<node_t, unsigned int> weights;
};

py::object kruskal_mst_edges(py::object G, py::object minimum, py::object weight, py::object data, py::object ignore_nan);
py::object prim_mst_edges(py::object G, py::object minimum, py::object weight, py::object data, py::object ignore_nan);
