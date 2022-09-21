#include "operation.h"
#include "graph.h"

py::object density(py::object G) {
    Graph& G_ = G.cast<Graph&>();
    int n = G_.node.size();
    int m = G.attr("number_of_edges")().cast<int>();
    if (m == 0 || n <= 1) {
        return py::cast(0);
    }
    weight_t d = m * 1.0 / (n * (n - 1));
    if(G.attr("is_directed")().equal(py::cast(false))){
        d*=2;
    }
    return py::cast(d);
}
