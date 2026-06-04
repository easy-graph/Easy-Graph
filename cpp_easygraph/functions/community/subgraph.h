#pragma once

#include <vector>
#include <memory>
#include "../../common/common.h"
#include "../../classes/graph.h"
#include "../../classes/directed_graph.h"
#include "../../classes/csr_graph.h"

py::object nodes_subgraph_cpp(py::object self, std::vector<node_t>& node_ids);