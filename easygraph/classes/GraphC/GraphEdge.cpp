#include "GraphMap.h"
#include "GraphEdge.h"

//作为sequence的方法
PyObject* GraphEdge_GetItem(GraphEdge* self, Py_ssize_t index) {
    PyObject* ret = nullptr;
    switch (index) {
    case 0:
        ret = PyDict_GetItem(self->id_to_node, PyLong_FromLong(self->edge.u));
        break;
    case 1:
        ret = PyDict_GetItem(self->id_to_node, PyLong_FromLong(self->edge.v));
        break;
    case 2: {
        GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
        temp_map->type = Msf;
        temp_map->pointer = self->edge.weight;
        ret = (PyObject*)temp_map;
        break;
    }
    default:
        PyErr_SetString(PyExc_IndexError, "invalid index");
    }
    return ret;
}

PySequenceMethods GraphEdge_sequence_methods = {
    nullptr,                               /* sq_length */
    nullptr,                               /* sq_concat */
    nullptr,                               /* sq_repeat */
    (ssizeargfunc)GraphEdge_GetItem,       /* sq_item */
    nullptr,                               /* was_sq_slice; */
    nullptr,                               /* sq_ass_item; */
    nullptr,                               /* was_sq_ass_slice */
    nullptr,                               /* sq_contains */
    nullptr,                               /* sq_inplace_concat */
    nullptr                                /* sq_inplace_repeat */
};

//内置方法
PyObject* GraphEdge_repr(GraphEdge* self) {
    GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
    temp_map->type = Msf;
    temp_map->pointer = self->edge.weight;
    return PyUnicode_FromFormat("(%R, %R, %R)", PyDict_GetItem(self->id_to_node, PyLong_FromLong(self->edge.u)), PyDict_GetItem(self->id_to_node, PyLong_FromLong(self->edge.v)), temp_map);
}

PyObject* GraphEdge_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    GraphEdge* self = (GraphEdge*)type->tp_alloc(type, 0);
    new (self)GraphEdge;
    return (PyObject*)self;
}

void* GraphEdge_dealloc(PyObject* obj) {
    Py_TYPE(obj)->tp_free(obj);
    return nullptr;
}

PyTypeObject GraphEdgeType = {
    PyObject_HEAD_INIT(NULL)
    "cpp_easygraph.GraphEdge",                         /* tp_name */
    sizeof(GraphEdge),                                 /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)GraphEdge_dealloc,                     /* tp_dealloc */
    0,                                                 /* tp_vectorcall_offset */
    nullptr,                                           /* tp_getattr */
    nullptr,                                           /* tp_setattr */
    nullptr,                                           /* tp_as_async */
    (reprfunc)GraphEdge_repr,                          /* tp_repr */
    nullptr,                                           /* tp_as_number */
    &GraphEdge_sequence_methods,                       /* tp_as_sequence */
    nullptr,                                           /* tp_as_mapping */
    nullptr,                                           /* tp_hash  */
    nullptr,                                           /* tp_call */
    (reprfunc)GraphEdge_repr,                          /* tp_str */
    nullptr,                                           /* tp_getattro */
    nullptr,                                           /* tp_setattro */
    nullptr,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                                /* tp_flags */
    "GraphEdge",                                       /* tp_doc */
    nullptr,                                           /* tp_traverse */
    nullptr,                                           /* tp_clear */
    nullptr,                                           /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    nullptr,                                           /* tp_iter */
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
    GraphEdge_new,                                     /* tp_new */
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