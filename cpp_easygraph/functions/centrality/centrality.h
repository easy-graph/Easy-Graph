#pragma once

#include "../../common/common.h"

py::object closeness_centrality(py::object G, py::object weight, py::object cutoff, py::object sources);
py::object betweenness_centrality(py::object G, py::object weight, py::object cutoff, py::object sources, 
                                    py::object normalized, py::object endpoints);
py::object cpp_katz_centrality(
    py::object G,
    py::object py_alpha,
    py::object py_beta,
    py::object py_max_iter,
    py::object py_tol,
    py::object py_normalized
);

py::object degree_centrality(py::object G);
py::object in_degree_centrality(py::object G);
py::object out_degree_centrality(py::object G);

py::object cpp_eigenvector_centrality(
    py::object G,
    py::object py_max_iter,
    py::object py_tol,
    py::object py_nstart,
    py::object py_weight
);