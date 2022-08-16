#pragma once

#include "../../common/common.h"

class NeighborIterator {
public:
    NeighborIterator() {
    }
    NeighborIterator(adj_attr_dict_factory& neighbor_map) {
        now = neighbor_map.begin();;
        end = neighbor_map.end();
    }
    node_t next() {
        if (now == end) {
            throw -1;
        }
        else {
            return (now++)->first;
        }
    }
private:
    adj_attr_dict_factory::iterator now, end;
};
typedef struct stackNode {
    node_t grandparent, parent;
    NeighborIterator neighbors_iter;
    stackNode(node_t grandparent, node_t parent, NeighborIterator neighbors_iter) {
        this->grandparent = grandparent;
        this->parent = parent;
        this->neighbors_iter = neighbors_iter;
    }
}stack_node;

py::object _biconnected_dfs_record_edges(py::object G, py::object need_components);
