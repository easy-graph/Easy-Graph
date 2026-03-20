#pragma once

#include "../common/common.h"

struct CSRGraph {
    std::vector<int> V;
    std::vector<int> E;
    std::vector<double> unweighted_W;
    std::unordered_map<std::string, std::shared_ptr<std::vector<double>>> W_map;

    std::vector<node_t> nodes;
    std::unordered_map<node_t, int> node2idx;

    // Oriented CSR cache used by triangle counting; invalidated on graph change
    bool oriented_valid = false;
    std::vector<int> rank_arr;
    std::vector<int>  order;
    std::vector<int>  oriented_V;
    std::vector<int>  oriented_E;
};