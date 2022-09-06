#pragma once

#include "../../common/common.h"

class UnionFind {
<<<<<<< HEAD
   public:
    UnionFind(py::object elements = py::none());
    node_t operator[](node_t node);
    void _union(node_t node1, node_t node2);
};

py::object cpp_prim_mst_edges(py::object G, py::object minimum, py::object weight, py::object data, py::object ignore_nan);
=======
public:
	UnionFind(std::vector<node_t> elements);
	node_t operator[](node_t object);
	void _union(node_t object1, node_t object2);

private:
	std::unordered_map<node_t, node_t> parents;
	std::unordered_map<node_t, unsigned int> weights;
};
>>>>>>> 7943fe837953dcfff3701edddb55528f23574b45
