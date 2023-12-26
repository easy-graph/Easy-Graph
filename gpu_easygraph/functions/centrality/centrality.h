#pragma once

#include <pybind11/pybind11.h>

#include "common.h"

/**
 * description: 
 *     The input and output accord with the definition in python function
 */
pybind11::object closeness_centrality (
    _IN_ pybind11::object py_G, 
    _IN_ pybind11::object py_weight, 
    _IN_ pybind11::object py_sources
);

/**
 * description: 
 *     The input and output accord with the definition in python function
 */
pybind11::object betweenness_centrality (
    _IN_ pybind11::object py_G, 
    _IN_ pybind11::object py_weight, 
    _IN_ pybind11::object py_sources,
    _IN_ pybind11::object py_normalized,
    _IN_ pybind11::object py_endpoints
);