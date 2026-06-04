#pragma once

#include <vector>
#include "../../classes/linkgraph.h"

std::vector<int> greedy_graph_coloring(const Graph_L& G);

std::vector<int> omp_graph_coloring(const Graph_L& G);

bool verify_coloring(const Graph_L& g, const std::vector<int>& colors);

int count_colors(const std::vector<int>& colors);