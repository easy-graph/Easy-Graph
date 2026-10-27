#pragma once

#include <cstdint>

namespace egdist {
using VertexId = std::int64_t;
using Offset = std::uint64_t;

struct EdgeRecord {
    VertexId source;
    VertexId target;
    double weight;
    Offset order;
};

struct Update {
    VertexId vertex;
    double value;
};
}
