#pragma once
#include "Graph.h"
#include "cmath"

PyObject* effective_size(PyObject* easygraph, PyObject* args, PyObject* kwargs);

PyObject* constraint(PyObject* easygraph, PyObject* args, PyObject* kwargs);

PyObject* hierarchy(PyObject* easygraph, PyObject* args, PyObject* kwargs);
