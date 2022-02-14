#include "GraphEdges.h"

PyObject* GraphEdgesIter_next(GraphEdgesIter* self) {
    if (self->iter != self->end) {
        GraphEdge* temp_edge = (GraphEdge*)PyObject_CallFunctionObjArgs((PyObject*)&GraphEdgeType, nullptr);
        temp_edge->id_to_node = self->id_to_node;
        temp_edge->node_to_id = self->node_to_id;
        temp_edge->edge = *((self->iter)++);
        return (PyObject*)temp_edge;
    }
    else {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
}

PyObject* GraphEdgesIter_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    GraphEdgesIter* self = (GraphEdgesIter*)type->tp_alloc(type, 0);
    new (self)GraphEdgesIter;
    return (PyObject*)self;
}

void* GraphEdgesIter_dealloc(PyObject* obj) {
    Py_TYPE(obj)->tp_free(obj);
    return nullptr;
}

PyTypeObject GraphEdgesIterType = {
    PyObject_HEAD_INIT(NULL)
    "cpp_easygraph.GraphEdgesIter",                    /* tp_name */
    sizeof(GraphEdgesIter),                            /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)GraphEdgesIter_dealloc,                /* tp_dealloc */
    0,                                                 /* tp_vectorcall_offset */
    nullptr,                                           /* tp_getattr */
    nullptr,                                           /* tp_setattr */
    nullptr,                                           /* tp_as_async */
    nullptr,                                           /* tp_repr */
    nullptr,                                           /* tp_as_number */
    nullptr,                                           /* tp_as_sequence */
    nullptr,                                           /* tp_as_mapping */
    nullptr,                                           /* tp_hash  */
    nullptr,                                           /* tp_call */
    nullptr,                                           /* tp_str */
    nullptr,                                           /* tp_getattro */
    nullptr,                                           /* tp_setattro */
    nullptr,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                                /* tp_flags */
    "GraphEdgesIter",                                  /* tp_doc */
    nullptr,                                           /* tp_traverse */
    nullptr,                                           /* tp_clear */
    nullptr,                                           /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    (getiterfunc)GraphEdgesIter_iter,                  /* tp_iter */
    (iternextfunc)GraphEdgesIter_next,                 /* tp_iternext */
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
    GraphEdgesIter_new,                                /* tp_new */
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

PyObject* GraphEdgesIter_iter(GraphEdgesIter* self) {
    GraphEdgesIter* temp_edges_iter = (GraphEdgesIter*)PyObject_CallFunctionObjArgs((PyObject*)&GraphEdgesIterType, nullptr);
    *temp_edges_iter = *self;
    return (PyObject*)temp_edges_iter;
}