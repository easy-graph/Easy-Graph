#pragma once

#include "../../common/common.h"


std::set<node_t> cpp_plain_bfs(py::object G,py::object source);
std::vector<std::set<node_t>> cpp_generator_connected_components(py::object G);
