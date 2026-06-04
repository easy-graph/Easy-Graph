#pragma once

#include <vector>
#include <unordered_set>
#include <string>

#include "../../classes/graph.h"

using namespace std;

py::object cpp_modularity(py::object G, py::object communities, py::object weight = py::str("weight"));
