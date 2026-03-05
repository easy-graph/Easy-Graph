#pragma once

#include "../../common/common.h"

py::object _dijkstra_multisource(py::object G, py::object sources, py::object weight, py::object target);
// py::object _dijkstra(py::object G, py::object sources, py::object weight, py::object target);
py::object _spfa(py::object G, py::object source, py::object weight);
py::object Floyd(py::object G,py::object weight);
py::object Prim(py::object G,py::object weight);
py::object Kruskal(py::object G,py::object weight);
py::object average_shortest_path_length(py::object G, py::object weight, py::object method);