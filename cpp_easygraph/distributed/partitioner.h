#pragma once
#include "types.h"

namespace egdist {
class Partitioner {
public:
    explicit Partitioner(int partitions);
    int owner(VertexId vertex) const;
private:
    int partitions_;
};
}
