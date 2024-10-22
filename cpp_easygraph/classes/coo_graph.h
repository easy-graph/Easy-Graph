#pragma once

#include "../common/common.h"

struct COOGraph {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<double> unweighted_W;
    std::unordered_map<std::string, std::shared_ptr<std::vector<double>>> W_map;

    std::vector<node_t> nodes;
    std::unordered_map<node_t, int> node2idx;
};