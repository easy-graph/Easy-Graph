#pragma once

#include "../../common/common.h"

py::object closeness_centrality(py::object G, py::object weight, py::object cutoff, py::object sources);
py::object betweenness_centrality(py::object G, py::object weight, py::object cutoff, py::object sources, 
                                    py::object normalized, py::object endpoints);