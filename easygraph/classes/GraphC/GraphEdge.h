#pragma once
#include "Common.h"

struct Edge {
    int u, v;
    std::map<std::string, float>* weight;
};

struct GraphEdge {
    PyObject_HEAD
        Edge edge;
    PyObject* node_to_id, * id_to_node;
};

//作为sequence的方法
PyObject* GraphEdge_GetItem(GraphEdge* self, Py_ssize_t index);

//内置方法
PyObject* GraphEdge_repr(GraphEdge* self);

PyObject* GraphEdge_new(PyTypeObject* type, PyObject* args, PyObject* kwds);

void* GraphEdge_dealloc(PyObject* obj);
