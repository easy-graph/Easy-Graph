#include "avg_degree.h"

#include "../../classes/graph.h"
py::object average_degree(py::object G) {
    Graph& G_ = G.cast<Graph&>();
    int n = G_.node.size();
    int m = G.attr("number_of_edges")().cast<int>();
    return py::cast(2.0 * m / n);
}