#pragma once
#include <memory>
#include <string>
#include "distributed_graph.h"

namespace egdist {
std::shared_ptr<DistributedGraph> read_edgelist(
    const std::string& path, std::shared_ptr<Communicator> comm, bool directed);
}
