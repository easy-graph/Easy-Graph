#include "mst.h"

UnionFind::UnionFind() {

}

UnionFind::UnionFind(std::vector<node_t> elements) {
	for (node_t x : elements) {
		parents[x] = x;
		weights[x] = 1;
	}
}

node_t UnionFind::operator[](node_t object) {
	if (!parents.count(object)) {
		parents[object] = object;
		weights[object] = 1;
		return object;
	}

	std::vector<node_t> path;
	path.push_back(object);
	node_t root = parents[object];
	while (root != path.back()) {
		path.push_back(root);
		root = parents[root];
	}
	for (node_t ancestor : path) {
		parents[ancestor] = root;
	}
	return root;
}

void UnionFind::_union(node_t object1, node_t object2) {
	node_t root, r;
	if (weights[object1] < weights[object2]) {
		root = object1, r = object2;
	}
	else {
		root = object2, r = object1;
	}
	weights[root] += weights[r];
	parents[r] = root;
}