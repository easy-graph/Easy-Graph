#pragma once
#include "Common.h"
#include "GraphEdge.h"

struct GraphEdgesIter {
    PyObject_HEAD
        std::vector<Edge>::iterator iter;
    std::vector<Edge>::iterator end;
    PyObject* node_to_id, * id_to_node;
};

PyObject* GraphEdgesIter_iter(GraphEdgesIter* self);

PyObject* GraphEdgesIter_next(GraphEdgesIter* self);

PyObject* GraphEdgesIter_new(PyTypeObject* type, PyObject* args, PyObject* kwds);

void* GraphEdgesIter_dealloc(PyObject* obj);

/*-----------------------------------------------------------------------------------------*/

struct GraphEdges {
    PyObject_HEAD
        std::vector<Edge> edges;
    PyObject* node_to_id, * id_to_node;
};

//作为sequence的方法
Py_ssize_t GraphEdges_len(GraphEdges* self);

PyObject* GraphEdges_GetItem(GraphEdges* self, Py_ssize_t index);

//内置方法
PyObject* GraphEdges_repr(GraphEdges* self);

PyObject* GraphEdges_iter(GraphEdges* self);

PyObject* GraphEdges_new(PyTypeObject* type, PyObject* args, PyObject* kwds);

void* GraphEdges_dealloc(PyObject* obj);