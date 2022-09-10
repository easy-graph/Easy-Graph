#pragma once

#include "../../common/common.h"


std::set<node_t> plain_bfs(py::object G,py::object source);
std::vector<std::set<node_t>> generator_connected_components(py::object G);
