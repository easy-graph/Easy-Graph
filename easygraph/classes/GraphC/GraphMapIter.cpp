#include "GraphMap.h"

PyMappingMethods GraphMapIter_mapping_methods = {
    nullptr,                               /* mp_length */
    nullptr,                               /* mp_subscript */
    nullptr,                               /* mp_ass_subscript */
};

PyObject* GraphMapIter_next(GraphMapIter* self) {
    switch (self->type) {
    case Msf: {
        if (self->Msf_iter != self->Msf_end) {
            if (self->flag == 1)
                return Py_BuildValue("f", ((self->Msf_iter)++)->second);
            else if (self->flag == 2) {
                auto temp_pointer = (self->Msf_iter)++;
                return Py_BuildValue("(sf)", (temp_pointer->first).c_str(), temp_pointer->second);
            }
            else
                return PyUnicode_FromString((((self->Msf_iter)++)->first).c_str());
        }
        else {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }
    }
    case Mif: {
        if (self->Mif_iter != self->Mif_end) {
            if (self->flag == 1)
                return Py_BuildValue("f", ((self->Mif_iter)++)->second);
            else if (self->flag == 2) {
                auto temp_pointer = (self->Mif_iter)++;
                return Py_BuildValue("(Of)", PyDict_GetItem(self->id_to_node, PyLong_FromLong(temp_pointer->first)), temp_pointer->second);
            }
            else
                return PyDict_GetItem(self->id_to_node, PyLong_FromLong(((self->Mif_iter)++)->first));
        }
        else {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }
    }
    case Mii: {
        if (self->Mii_iter != self->Mii_end) {
            if (self->flag == 1)
                return PyLong_FromLong(((self->Mii_iter)++)->second);
            else if (self->flag == 2) {
                auto temp_pointer = (self->Mii_iter)++;
                return Py_BuildValue("(Oi)", PyDict_GetItem(self->id_to_node, PyLong_FromLong(temp_pointer->first)), temp_pointer->second);
            }
            else
                return PyDict_GetItem(self->id_to_node, PyLong_FromLong(((self->Mii_iter)++)->first));
        }
        else {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }
    }
    case MiMsf: {
        if (self->MiMsf_iter != self->MiMsf_end) {
            if (self->flag == 1) {
                GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
                temp_map->type = Msf;
                temp_map->pointer = &(((self->MiMsf_iter)++)->second);
                temp_map->id_to_node = self->id_to_node;
                temp_map->node_to_id = self->node_to_id;
                return (PyObject*)temp_map;
            }
            else if (self->flag == 2) {
                auto temp_pointer = (self->MiMsf_iter)++;
                GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
                temp_map->type = Msf;
                temp_map->pointer = &(temp_pointer->second);
                temp_map->id_to_node = self->id_to_node;
                temp_map->node_to_id = self->node_to_id;
                return Py_BuildValue("(OO)", PyDict_GetItem(self->id_to_node, PyLong_FromLong(temp_pointer->first)), temp_map);
            }
            else
                return PyDict_GetItem(self->id_to_node, PyLong_FromLong(((self->MiMsf_iter)++)->first));
        }
        else {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }
    }
    case MiMiMsf: {
        if (self->MiMiMsf_iter != self->MiMiMsf_end) {
            if (self->flag == 1) {
                GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
                temp_map->type = MiMsf;
                temp_map->pointer = &(((self->MiMiMsf_iter)++)->second);
                temp_map->id_to_node = self->id_to_node;
                temp_map->node_to_id = self->node_to_id;
                return (PyObject*)temp_map;
            }
            else if (self->flag == 2) {
                auto temp_pointer = (self->MiMiMsf_iter)++;
                GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
                temp_map->type = MiMsf;
                temp_map->pointer = &(temp_pointer->second);
                temp_map->id_to_node = self->id_to_node;
                temp_map->node_to_id = self->node_to_id;
                return Py_BuildValue("(OO)", PyDict_GetItem(self->id_to_node, PyLong_FromLong(temp_pointer->first)), temp_map);
            }
            else
                return PyDict_GetItem(self->id_to_node, PyLong_FromLong(((self->MiMiMsf_iter)++)->first));
        }
        else {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }
    }
    }
}

PyObject* GraphMapIter_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    GraphMapIter* self = (GraphMapIter*)type->tp_alloc(type, 0);
    new (self)GraphMapIter;
    return (PyObject*)self;
}

void* GraphMapIter_dealloc(PyObject* obj) {
    Py_TYPE(obj)->tp_free(obj);
    return nullptr;
}

PyTypeObject GraphMapIterType = {
    PyObject_HEAD_INIT(NULL)
    "cpp_easygraph.GraphMapIter",                      /* tp_name */
    sizeof(GraphMapIter),                              /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)GraphMapIter_dealloc,                  /* tp_dealloc */
    0,                                                 /* tp_vectorcall_offset */
    nullptr,                                           /* tp_getattr */
    nullptr,                                           /* tp_setattr */
    nullptr,                                           /* tp_as_async */
    nullptr,                                           /* tp_repr */
    nullptr,                                           /* tp_as_number */
    nullptr,                                           /* tp_as_sequence */
    &GraphMapIter_mapping_methods,                     /* tp_as_mapping */
    nullptr,                                           /* tp_hash  */
    nullptr,                                           /* tp_call */
    nullptr,                                           /* tp_str */
    nullptr,                                           /* tp_getattro */
    nullptr,                                           /* tp_setattro */
    nullptr,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                                /* tp_flags */
    "GraphMapIter",                                    /* tp_doc */
    nullptr,                                           /* tp_traverse */
    nullptr,                                           /* tp_clear */
    nullptr,                                           /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    (getiterfunc)GraphMapIter_iter,                    /* tp_iter */
    (iternextfunc)GraphMapIter_next,                   /* tp_iternext */
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
    GraphMapIter_new,                                  /* tp_new */
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

PyObject* GraphMapIter_iter(GraphMapIter* self) {
    GraphMapIter* temp_map_iter = (GraphMapIter*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapIterType, nullptr);
    *temp_map_iter = *self;
    return (PyObject*)temp_map_iter;
}


