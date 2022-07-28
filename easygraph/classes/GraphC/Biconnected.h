#pragma once
#define BOOST_PYTHON_STATIC_LIB

#include "Common.h"
#include "Graph.h"

class NeighborIterator {
public:
    NeighborIterator() {
    }
    NeighborIterator(Graph::adj_attr_dict_factory& neighbor_map) {
        now = neighbor_map.begin();;
        end = neighbor_map.end();
    }
    Graph::node_t next() throw(...) {
        if (now == end) {
            throw - 1;
        }
        else {
            return (now++)->first;
        }
    }
private:
    Graph::adj_attr_dict_factory::iterator now, end;
};
typedef struct stackNode {
    Graph::node_t grandparent, parent;
    NeighborIterator neighbors_iter;
    stackNode(Graph::node_t grandparent, Graph::node_t parent, NeighborIterator neighbors_iter) {
        this->grandparent = grandparent;
        this->parent = parent;
        this->neighbors_iter = neighbors_iter;
    }
}stack_node;

py::object generator_biconnected_components_edges(py::object G);
