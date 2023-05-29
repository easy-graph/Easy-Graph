#pragma once

#include "../../common/common.h"

py::object strongly_connected_components(py::object G);
py::object strongly_connected_components_iteration_impl(py::object G);