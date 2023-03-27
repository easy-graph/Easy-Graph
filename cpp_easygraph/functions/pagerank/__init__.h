#pragma once

#include "../../common/common.h"

py::object _pagerank(py::object G, double alpha, int max_iterator, double threshold);