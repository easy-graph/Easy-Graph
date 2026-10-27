#include "distributed_graph.h"

#include <algorithm>
#include <utility>

namespace egdist {
DistributedGraph::DistributedGraph(std::shared_ptr<Communicator> comm_, bool directed_,
                                  std::vector<VertexId> owned, std::vector<EdgeRecord> edges)
    : comm(std::move(comm_)), partitioner(comm->size()), directed(directed_) {
    std::sort(owned.begin(), owned.end());
    owned.erase(std::unique(owned.begin(), owned.end()), owned.end());
    vertices = std::move(owned);
    for (std::size_t i = 0; i < vertices.size(); ++i) local_index.emplace(vertices[i], i);
    std::sort(edges.begin(), edges.end(), [](const EdgeRecord& a, const EdgeRecord& b) {
        if (a.source != b.source) return a.source < b.source;
        if (a.target != b.target) return a.target < b.target;
        return a.order > b.order;
    });
    edges.erase(std::unique(edges.begin(), edges.end(), [](const EdgeRecord& a, const EdgeRecord& b) {
        return a.source == b.source && a.target == b.target;
    }), edges.end());
    offsets.assign(vertices.size() + 1, 0);
    Offset logical_edges = 0;
    for (const auto& edge : edges) {
        ++offsets[local_index.at(edge.source) + 1];
        targets.push_back(edge.target);
        weights.push_back(edge.weight);
        if (directed || edge.source <= edge.target) ++logical_edges;
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
    const int rank = comm->rank();
    remote_targets.resize(comm->size());
    for (auto target : targets) {
        const int owner = partitioner.owner(target);
        if (owner != rank) remote_targets[owner].push_back(target);
    }
    std::unordered_map<VertexId, std::size_t> remote_slots;
    contribution_slots = vertices.size();
    for (auto& bucket : remote_targets) {
        std::sort(bucket.begin(), bucket.end());
        bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
        for (auto target : bucket) remote_slots.emplace(target, contribution_slots++);
    }
    target_slots.reserve(targets.size());
    for (auto target : targets) {
        target_slots.push_back(partitioner.owner(target) == rank ?
                               local_index.at(target) : remote_slots.at(target));
    }
    global_num_vertices = comm->sum(static_cast<Offset>(vertices.size()));
    global_num_edges = comm->sum(logical_edges);
}
}
