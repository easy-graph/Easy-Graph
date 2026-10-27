#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "communicator.h"
#include "partitioner.h"
#include "types.h"

namespace egdist {
class DistributedGraph {
public:
    DistributedGraph(std::shared_ptr<Communicator> comm, bool directed,
                     std::vector<VertexId> owned, std::vector<EdgeRecord> edges);
    std::shared_ptr<Communicator> comm;
    Partitioner partitioner;
    bool directed;
    Offset global_num_vertices = 0;
    Offset global_num_edges = 0;
    std::vector<VertexId> vertices;
    std::unordered_map<VertexId, std::size_t> local_index;
    std::vector<Offset> offsets;

    std::vector<VertexId> targets;
    std::vector<double> weights;
    std::vector<std::size_t> target_slots;
    std::vector<std::vector<VertexId>> remote_targets;
    std::size_t contribution_slots = 0;
};
}
