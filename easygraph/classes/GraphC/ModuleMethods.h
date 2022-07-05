#pragma once
#include "Graph.h"

PyObject* effective_size(PyObject* easygraph, PyObject* args, PyObject* kwargs);

PyObject* constraint(PyObject* easygraph, PyObject* args, PyObject* kwargs);

PyObject* hierarchy(PyObject* easygraph, PyObject* args, PyObject* kwargs);

PyObject* _dijkstra_multisource(PyObject* easygraph, PyObject* args, PyObject* kwargs);

PyObject* _biconnected_dfs_record_edges(PyObject* easygraph, PyObject* args, PyObject* kwargs);