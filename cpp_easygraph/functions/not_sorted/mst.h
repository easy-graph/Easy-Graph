#pragma once

#include "../../common/common.h"

class UnionFind {
public:
	UnionFind(py::object elements = py::none());
	node_t operator[](node_t node);
	void _union(node_t node1, node_t node2);
};