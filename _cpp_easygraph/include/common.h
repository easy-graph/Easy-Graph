#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <fstream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <set>
#include <cmath>
#include <algorithm>
#include <queue>
#include <vector>

namespace py = pybind11;

using node_t = int;
using weight_t = float;
using node_attr_dict_factory =
    std::map<std::string, weight_t>; //(weight_key, value)
using edge_attr_dict_factory =
    std::map<std::string, weight_t>; //(weight_key, value)
using node_dict_factory =
    std::unordered_map<node_t, node_attr_dict_factory>; //(node, node_attr)
using adj_attr_dict_factory =
    std::unordered_map<node_t, edge_attr_dict_factory>; //(out_node, (weight_key, value))
using adj_dict_factory =
    std::unordered_map<node_t, adj_attr_dict_factory>; //(node, edge_attr)
