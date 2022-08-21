#include "operation.h"
#include "graph.h"

py::object density(py::object G) {
    Graph& G_ = G.cast<Graph&>();
    int n = G_.node.size();
    adj_dict_factory adj = G_.adj;
    int m = 0;
    for (adj_dict_factory::iterator i = adj.begin(); i != adj.end(); i++) {
        adj_attr_dict_factory node_edge = i->second;
        m += node_edge.size();
    }
    if (m == 0 || n <= 1) {
        return py::cast(0);
    }
    weight_t d = m * 1.0 / (n * (n - 1));
    return py::cast(d);
}
