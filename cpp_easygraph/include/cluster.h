#pragma once


#include "common.h"

py::object clustering(py::object G,
					  py::object nodes = py::none(),
					  py::object weight = py::none());
