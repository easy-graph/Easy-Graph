#include "GraphEdges.h"

//作为sequence的方法
Py_ssize_t GraphEdges_len(GraphEdges* self) {
    return self->edges.size();
}

PyObject* GraphEdges_GetItem(GraphEdges* self, Py_ssize_t index) {
    if (index >= self->edges.size()) {
        PyErr_SetString(PyExc_IndexError, "invalid index");
        return nullptr;
    }
    else {
        GraphEdge* ret = (GraphEdge*)PyObject_CallFunctionObjArgs((PyObject*)&GraphEdgeType, nullptr);
        ret->id_to_node = self->id_to_node;
        ret->node_to_id = self->node_to_id;
        ret->edge = self->edges[index];
        return (PyObject*)ret;
    }
}

PySequenceMethods GraphEdges_sequence_methods = {
    (lenfunc)GraphEdges_len,               /* sq_length */
    nullptr,                               /* sq_concat */
    nullptr,                               /* sq_repeat */
    (ssizeargfunc)GraphEdges_GetItem,      /* sq_item */
    nullptr,                               /* was_sq_slice; */
    nullptr,                               /* sq_ass_item; */
    nullptr,                               /* was_sq_ass_slice */
    nullptr,                               /* sq_contains */
    nullptr,                               /* sq_inplace_concat */
    nullptr                                /* sq_inplace_repeat */
};

//内置方法
PyObject* GraphEdges_repr(GraphEdges* self) {
    PyObject* ret = PyUnicode_FromString("[");
    auto iter = self->edges.begin(), end = self->edges.end();
    GraphEdge* temp_edge = (GraphEdge*)PyObject_CallFunctionObjArgs((PyObject*)&GraphEdgeType, nullptr);
    temp_edge->node_to_id = self->node_to_id;
    temp_edge->id_to_node = self->id_to_node;
    if (iter != end) {
        temp_edge->edge = *iter;
        ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("%R", temp_edge));
        iter++;
    }
    while (iter != end) {
        temp_edge->edge = *iter;
        ret = PyUnicode_Concat(ret, PyUnicode_FromFormat(", %R", temp_edge));
        iter++;
    }
    ret = PyUnicode_Concat(ret, PyUnicode_FromString("]"));
    return ret;
}

PyObject* GraphEdges_iter(GraphEdges* self) {
    GraphEdgesIter* temp_iter = (GraphEdgesIter*)PyObject_CallFunctionObjArgs((PyObject*)&GraphEdgesIterType, nullptr);
    temp_iter->id_to_node = self->id_to_node;
    temp_iter->node_to_id = self->node_to_id;
    temp_iter->iter = self->edges.begin();
    temp_iter->end = self->edges.end();
    return (PyObject*)temp_iter;
}

PyObject* GraphEdges_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    GraphEdges* self = (GraphEdges*)type->tp_alloc(type, 0);
    new (self)GraphEdges;
    return (PyObject*)self;
}

void* GraphEdges_dealloc(PyObject* obj) {
    Py_TYPE(obj)->tp_free(obj);
    return nullptr;
}

PyTypeObject GraphEdgesType = {
    PyObject_HEAD_INIT(NULL)
    "cpp_easygraph.GraphEdges",                        /* tp_name */
    sizeof(GraphEdges),                                /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)GraphEdges_dealloc,                    /* tp_dealloc */
    0,                                                 /* tp_vectorcall_offset */
    nullptr,                                           /* tp_getattr */
    nullptr,                                           /* tp_setattr */
    nullptr,                                           /* tp_as_async */
    (reprfunc)GraphEdges_repr,                         /* tp_repr */
    nullptr,                                           /* tp_as_number */
    &GraphEdges_sequence_methods,                      /* tp_as_sequence */
    nullptr,                                           /* tp_as_mapping */
    nullptr,                                           /* tp_hash  */
    nullptr,                                           /* tp_call */
    (reprfunc)GraphEdges_repr,                         /* tp_str */
    nullptr,                                           /* tp_getattro */
    nullptr,                                           /* tp_setattro */
    nullptr,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                                /* tp_flags */
    "GraphEdges",                                      /* tp_doc */
    nullptr,                                           /* tp_traverse */
    nullptr,                                           /* tp_clear */
    nullptr,                                           /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    (getiterfunc)GraphEdges_iter,                      /* tp_iter */
    nullptr,                                           /* tp_iternext */
    nullptr,                                           /* tp_methods */
    nullptr,                                           /* tp_members */
    nullptr,                                           /* tp_getset */
    nullptr,                                           /* tp_base */
    nullptr,                                           /* tp_dict */
    nullptr,                                           /* tp_descr_get */
    nullptr,                                           /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    nullptr,                                           /* tp_init */
    nullptr,                                           /* tp_alloc */
    GraphEdges_new,                                     /* tp_new */
    nullptr,                                           /* tp_free */
    nullptr,                                           /* tp_is_gc */
    nullptr,                                           /* tp_bases */
    nullptr,                                           /* tp_mro */
    nullptr,                                           /* tp_cache */
    nullptr,                                           /* tp_subclasses */
    nullptr,                                           /* tp_weaklist */
    nullptr,                                           /* tp_del */
    0,                                                 /* tp_version_tag */
    nullptr,                                           /* tp_finalize */
#if PY_VERSION_HEX >= 0x03080000
    nullptr,                                           /* tp_vectorcall */
#endif
};
