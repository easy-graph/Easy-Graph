#include "partitioner.h"
#include <stdexcept>

namespace egdist {
Partitioner::Partitioner(int partitions) : partitions_(partitions) {
    if (partitions <= 0) throw std::invalid_argument("Partition count must be positive");
}
int Partitioner::owner(VertexId vertex) const {
    if (vertex < 0) throw std::invalid_argument("Vertex IDs must be nonnegative");
    return static_cast<int>(vertex % partitions_);
}
}
