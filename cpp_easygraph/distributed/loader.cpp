#include "loader.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace egdist {
namespace {
VertexId parse_id(const std::string& token) {
    std::size_t end = 0;
    const auto value = std::stoll(token, &end);
    if (end != token.size() || value < 0) {
        throw std::invalid_argument("Vertex IDs must be nonnegative signed 64-bit integers");
    }
    return static_cast<VertexId>(value);
}
}

std::shared_ptr<DistributedGraph> read_edgelist(
    const std::string& path, std::shared_ptr<Communicator> comm, bool directed) {
    comm->agree(path);
    comm->agree(directed ? "directed" : "undirected");
    std::ifstream stream(path, std::ios::binary);
    comm->check_error(stream ? "" : "Cannot open edge list: " + path);
    stream.seekg(0, std::ios::end);
    const auto file_end = stream.tellg();
    comm->check_error(file_end < 0 ? "Cannot determine edge-list file size" : "");
    const auto bytes = static_cast<Offset>(file_end);
    comm->agree(std::to_string(bytes));

    const auto rank = comm->rank();
    const auto size = comm->size();
    const Offset begin = bytes / size * rank + std::min<Offset>(bytes % size, rank);
    const Offset end = bytes / size * (rank + 1) + std::min<Offset>(bytes % size, rank + 1);
    stream.seekg(static_cast<std::streamoff>(begin));
    if (begin > 0) {
        stream.seekg(static_cast<std::streamoff>(begin - 1));
        char previous = 0;
        stream.get(previous);
        if (previous != '\n') {
            std::string partial;
            std::getline(stream, partial);
        }
    }
    Partitioner partitioner(size);
    std::vector<std::vector<VertexId>> vertex_buckets(size);
    std::vector<std::vector<EdgeRecord>> edge_buckets(size);
    std::string error;
    while (stream) {
        const auto position = stream.tellg();
        if (position < 0 || static_cast<Offset>(position) >= end) break;
        std::string line;
        if (!std::getline(stream, line)) break;
        line = line.substr(0, line.find('#'));
        std::istringstream record(line);
        std::vector<std::string> tokens;
        for (std::string token; record >> token;) tokens.push_back(token);
        if (tokens.empty()) continue;
        try {
            if (tokens.size() > 3) throw std::invalid_argument("Expected u, u v, or u v weight");
            const auto u = parse_id(tokens[0]);
            vertex_buckets[partitioner.owner(u)].push_back(u);
            if (tokens.size() == 1) continue;
            const auto v = parse_id(tokens[1]);
            double weight = 1.0;
            if (tokens.size() == 3) {
                std::size_t parsed = 0;
                weight = std::stod(tokens[2], &parsed);
                if (parsed != tokens[2].size() || !std::isfinite(weight) || weight < 0) {
                    throw std::invalid_argument("Weights must be finite and nonnegative");
                }
            }
            vertex_buckets[partitioner.owner(v)].push_back(v);
            const auto order = static_cast<Offset>(position);
            edge_buckets[partitioner.owner(u)].push_back({u, v, weight, order});
            if (!directed && u != v) {
                edge_buckets[partitioner.owner(v)].push_back({v, u, weight, order});
            }
        } catch (const std::exception& exc) {
            error = "Invalid record at byte " + std::to_string(static_cast<Offset>(position)) +
                    ": " + exc.what();
            break;
        }
    }
    if (stream.bad()) error = "I/O error while reading edge list: " + path;
    comm->check_error(error);

    for (auto& bucket : vertex_buckets) {
        std::sort(bucket.begin(), bucket.end());
        bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
    }
    auto vertices = comm->exchange(vertex_buckets);
    auto edges = comm->exchange(edge_buckets);
    return std::make_shared<DistributedGraph>(comm, directed, std::move(vertices), std::move(edges));
}
}
