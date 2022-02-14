#include "GraphMap.h"

PyMethodDef GraphMapMethods[] = {
    {"keys", (PyCFunction)GraphMap_keys, METH_VARARGS | METH_KEYWORDS, "" },
    {"copy", (PyCFunction)GraphMap_copy, METH_VARARGS | METH_KEYWORDS, "" },
    {"get", (PyCFunction)GraphMap_get, METH_VARARGS | METH_KEYWORDS, "" },
    {"values", (PyCFunction)GraphMap_values, METH_VARARGS | METH_KEYWORDS, "" },
    {"items", (PyCFunction)GraphMap_items, METH_VARARGS | METH_KEYWORDS, "" },
    {NULL}
};

//作为mapping的方法
Py_ssize_t GraphMap_len(GraphMap* self) {
    Py_ssize_t ret = 0;
    switch (self->type) {
    case Msf: {
        ret = ((std::unordered_map<std::string, float>*)(self->pointer))->size();
        break;
    }
    case Mif: {
        ret = ((std::unordered_map<int, float>*)(self->pointer))->size();
        break;
    }
    case Mii: {
        ret = ((std::unordered_map<int, int>*)(self->pointer))->size();
        break;
    }
    case MiMsf: {
        ret = ((std::unordered_map<int, std::unordered_map<std::string, float>>*)(self->pointer))->size();
        break;
    }
    case MiMiMsf: {
        ret = ((std::unordered_map<int, std::unordered_map<int, std::unordered_map<std::string, float>>>*)(self->pointer))->size();
        break;
    }
    }
    return ret;
}

PyMappingMethods GraphMap_mapping_methods = {
    (lenfunc)GraphMap_len,                 /* mp_length */
    (binaryfunc)GraphMap_getitem,          /* mp_subscript */
    nullptr,                               /* mp_ass_subscript */
};

//作为sequence的方法
int GraphMap_contains(GraphMap* self, PyObject* args) {
    int ret = 0;
    switch (self->type) {
    case Msf: {
        std::string weight;
        PyObject* weight_unicode;
        PyArg_Parse(args, "O", &weight_unicode);
        weight = PyUnicode_AsUTF8(weight_unicode);
        ret = ((std::unordered_map<std::string, float>*)(self->pointer))->count(weight);
        break;
    }
    case Mif: {
        PyObject* pnode = PyTuple_GetItem(args, 0);
        int nodeid = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pnode));
        ret = ((std::unordered_map<int, float>*)(self->pointer))->count(nodeid);
        break;
    }
    case Mii: {
        PyObject* pnode = PyTuple_GetItem(args, 0);
        int nodeid = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pnode));
        ret = ((std::unordered_map<int, int>*)(self->pointer))->count(nodeid);
        break;
    }
    case MiMsf: {
        PyObject* pnode = PyTuple_GetItem(args, 0);
        int nodeid = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pnode));
        ret = ((std::unordered_map<int, std::unordered_map<std::string, float>>*)(self->pointer))->count(nodeid);
        break;
    }
    case MiMiMsf: {
        PyObject* pnode = PyTuple_GetItem(args, 0);
        int nodeid = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pnode));
        ret = ((std::unordered_map<int, std::unordered_map<int, std::unordered_map<std::string, float>>>*)(self->pointer))->count(nodeid);
        break;
    }
    }
    return ret;
}

PySequenceMethods GraphMap_sequence_methods = {
    nullptr,                               /* sq_length */
    nullptr,                               /* sq_concat */
    nullptr,                               /* sq_repeat */
    nullptr,                               /* sq_item */
    nullptr,                               /* was_sq_slice; */
    nullptr,                               /* sq_ass_item; */
    nullptr,                               /* was_sq_ass_slice */
    (objobjproc)GraphMap_contains,         /* sq_contains */
    nullptr,                               /* sq_inplace_concat */
    nullptr                                /* sq_inplace_repeat */
};

//内置方法
PyObject* GraphMap_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    GraphMap* self = (GraphMap*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

void* GraphMap_dealloc(PyObject* obj) {
    GraphMap* self = (GraphMap*)obj;
    if (self->flag) {
        switch (self->type) {
        case Msf: {
            std::unordered_map<std::string, float>* for_delete = (std::unordered_map<std::string, float>*)(self->pointer);
            delete for_delete;
            break;
        }
        case Mif: {
            std::unordered_map<int, float>* for_delete = (std::unordered_map<int, float>*)(self->pointer);
            delete for_delete;
            break;
        }
        case Mii: {
            std::unordered_map<int, int>* for_delete = (std::unordered_map<int, int>*)(self->pointer);
            delete for_delete;
            break;
        }
        case MiMsf: {
            std::unordered_map<int, std::unordered_map<std::string, float>>* for_delete = (std::unordered_map<int, std::unordered_map<std::string, float>>*)(self->pointer);
            delete for_delete;
            break;
        }
        case MiMiMsf: {
            std::unordered_map<int, std::unordered_map<int, std::unordered_map<std::string, float>>>* for_delete = (std::unordered_map<int, std::unordered_map<int, std::unordered_map<std::string, float>>>*)(self->pointer);
            delete for_delete;
            break;
        }
        }
    }
    Py_TYPE(obj)->tp_free(obj);
    return nullptr;
}

PyObject* GraphMap_iter(GraphMap* self) {
    GraphMapIter* temp_iter = (GraphMapIter*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapIterType, nullptr);
    temp_iter->type = self->type;
    temp_iter->id_to_node = self->id_to_node;
    temp_iter->node_to_id = self->node_to_id;
    switch (self->type) {
    case Msf: {
        temp_iter->Msf_iter = ((std::map<std::string, float>*)(self->pointer))->begin();
        temp_iter->Msf_end = ((std::map<std::string, float>*)(self->pointer))->end();
        break;
    }
    case Mif: {
        temp_iter->Mif_iter = ((std::unordered_map<int, float>*)(self->pointer))->begin();
        temp_iter->Mif_end = ((std::unordered_map<int, float>*)(self->pointer))->end();
        break;
    }
    case Mii: {
        temp_iter->Mii_iter = ((std::unordered_map<int, int>*)(self->pointer))->begin();
        temp_iter->Mii_end = ((std::unordered_map<int, int>*)(self->pointer))->end();
        break;
    }
    case MiMsf: {
        temp_iter->MiMsf_iter = ((std::unordered_map<int, std::map<std::string, float>>*)(self->pointer))->begin();
        temp_iter->MiMsf_end = ((std::unordered_map<int, std::map<std::string, float>>*)(self->pointer))->end();
        break;
    }
    case MiMiMsf: {
        temp_iter->MiMiMsf_iter = ((std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>*)(self->pointer))->begin();
        temp_iter->MiMiMsf_end = ((std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>*)(self->pointer))->end();
        break;
    }
    }
    return (PyObject*)temp_iter;
}

PyTypeObject GraphMapType = {
    PyObject_HEAD_INIT(NULL)
    "cpp_easygraph.GraphMap",                          /* tp_name */
    sizeof(GraphMap),                                  /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)GraphMap_dealloc,                      /* tp_dealloc */
    0,                                                 /* tp_vectorcall_offset */
    nullptr,                                           /* tp_getattr */
    nullptr,                                           /* tp_setattr */
    nullptr,                                           /* tp_as_async */
    (reprfunc)GraphMap_repr,                           /* tp_repr */
    nullptr,                                           /* tp_as_number */
    &GraphMap_sequence_methods,                        /* tp_as_sequence */
    &GraphMap_mapping_methods,                         /* tp_as_mapping */
    nullptr,                                           /* tp_hash  */
    nullptr,                                           /* tp_call */
    (reprfunc)GraphMap_repr,                           /* tp_str */
    nullptr,                                           /* tp_getattro */
    nullptr,                                           /* tp_setattro */
    nullptr,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                                /* tp_flags */
    "GraphMap",                                        /* tp_doc */
    nullptr,                                           /* tp_traverse */
    nullptr,                                           /* tp_clear */
    nullptr,                                           /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    (getiterfunc)GraphMap_iter,                        /* tp_iter */
    nullptr,                                           /* tp_iternext */
    GraphMapMethods,                                   /* tp_methods */
    nullptr,                                           /* tp_members */
    nullptr,                                           /* tp_getset */
    nullptr,                                           /* tp_base */
    nullptr,                                           /* tp_dict */
    nullptr,                                           /* tp_descr_get */
    nullptr,                                           /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    nullptr,                                           /* tp_init */
    nullptr,                                           /* tp_alloc */
    GraphMap_new,                                      /* tp_new */
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

//下面是部分需要额外放置的函数体
PyObject* _GraphMap_getitem(GraphMap* self, PyObject* pykey, PyObject* default_val = Py_None) {
    PyObject* ret = Py_None;
    switch (self->type) {
    case Msf: {
        if (pykey == Py_None) {
            return PyLong_FromLong(PyLong_AsLong(default_val));
        }
        std::string key(PyUnicode_AsUTF8(pykey));
        std::map<std::string, float>* m = (std::map<std::string, float>*)(self->pointer);
        if (m->count(key))
            ret = Py_BuildValue("f", (*m)[key]);
        break;
    }
    case Mif: {
        int key = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pykey));
        std::unordered_map<int, float>* m = (std::unordered_map<int, float>*)(self->pointer);
        if (m->count(key))
            ret = Py_BuildValue("f", (*m)[key]);
        break;
    }
    case Mii: {
        int key = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pykey));
        std::unordered_map<int, int>* m = (std::unordered_map<int, int>*)(self->pointer);
        if (m->count(key))
            ret = Py_BuildValue("i", (*m)[key]);
        break;
    }
    case MiMsf: {
        int key = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pykey));
        GraphMap* m = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
        m->id_to_node = self->id_to_node;
        m->node_to_id = self->node_to_id;
        auto temp_pointer = (std::unordered_map<int, std::map<std::string, float>>*)self->pointer;
        if (temp_pointer->count(key)) {
            m->pointer = &((*temp_pointer)[key]);
            m->type = Msf;
            ret = (PyObject*)m;
        }
        break;
    }
    case MiMiMsf: {
        int key = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pykey));
        GraphMap* m = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
        m->id_to_node = self->id_to_node;
        m->node_to_id = self->node_to_id;
        auto temp_pointer = (std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>*)self->pointer;
        if (temp_pointer->count(key)) {
            m->pointer = &((*temp_pointer)[key]);
            m->type = MiMsf;
            ret = (PyObject*)m;
        }
        break;
    }
    }
    return ret;
}

PyObject* GraphMap_getitem(GraphMap* self, PyObject* pykey) {
    PyObject* ret = _GraphMap_getitem(self, pykey);
    if (!ret) {
        if (self->type == Msf) {
            PyErr_Format(PyExc_KeyError, "No weight named %R.", pykey);
        }
        else {
            PyErr_Format(PyExc_KeyError, "No node named %R.", pykey);
        }
    }
    return ret;
}

PyObject* GraphMap_repr(GraphMap* self) {
    PyObject* ret = PyUnicode_FromString("{");
    switch (self->type) {
    case Msf: {
        std::map<std::string, float>* m = (std::map<std::string, float>*)(self->pointer);
        auto iter = m->begin(), end = m->end();
        if (iter != end) {
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("\'%s\': %R", iter->first.c_str(), PyFloat_FromDouble(iter->second)));
            iter++;
        }
        while (iter != end) {
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat(",\' %s\': %R,", iter->first.c_str(), PyFloat_FromDouble(iter->second)));
            iter++;
        }
        ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("}"));
        break;
    }
    case Mif: {
        std::unordered_map<int, float>* m = (std::unordered_map<int, float>*)(self->pointer);
        auto iter = m->begin(), end = m->end();
        if (iter != end) {
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("%R: %R", PyDict_GetItem(self->id_to_node, PyLong_FromLong(iter->first)), PyFloat_FromDouble(iter->second)));
            iter++;
        }
        while (iter != end) {
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat(", %R: %R", PyDict_GetItem(self->id_to_node, PyLong_FromLong(iter->first)), PyFloat_FromDouble(iter->second)));
            iter++;
        }
        ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("}"));
        break;
    }
    case Mii: {
        std::unordered_map<int, int>* m = (std::unordered_map<int, int>*)(self->pointer);
        auto iter = m->begin(), end = m->end();
        if (iter != end) {
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("%R: %d", PyDict_GetItem(self->id_to_node, PyLong_FromLong(iter->first)), iter->second));
            iter++;
        }
        while (iter != end) {
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat(", %R: %d", PyDict_GetItem(self->id_to_node, PyLong_FromLong(iter->first)), iter->second));
            iter++;
        }
        ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("}"));
        break;
    }
    case MiMsf: {
        std::unordered_map<int, std::map<std::string, float>>* m = (std::unordered_map<int, std::map<std::string, float>>*)(self->pointer);
        auto iter = m->begin(), end = m->end();
        GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
        temp_map->type = Msf;
        temp_map->id_to_node = self->id_to_node;
        temp_map->node_to_id = self->node_to_id;
        if (iter != end) {
            temp_map->pointer = &(iter->second);
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("%R: %R", PyDict_GetItem(self->id_to_node, PyLong_FromLong(iter->first)), (PyObject*)temp_map));
            iter++;
        }
        while (iter != end) {
            temp_map->pointer = &(iter->second);
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat(", %R: %R", PyDict_GetItem(self->id_to_node, PyLong_FromLong(iter->first)), (PyObject*)temp_map));
            iter++;
        }
        ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("}"));
        Py_DecRef((PyObject*)temp_map);
        break;
    }
    case MiMiMsf: {
        std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>* m = (std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>*)(self->pointer);
        auto iter = m->begin(), end = m->end();
        GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
        temp_map->type = MiMsf;
        temp_map->id_to_node = self->id_to_node;
        temp_map->node_to_id = self->node_to_id;
        if (iter != end) {
            temp_map->pointer = &(iter->second);
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("%R: %R", PyDict_GetItem(self->id_to_node, PyLong_FromLong(iter->first)), (PyObject*)temp_map));
            iter++;
        }
        while (iter != end) {
            temp_map->pointer = &(iter->second);
            ret = PyUnicode_Concat(ret, PyUnicode_FromFormat(", %R: %R", PyDict_GetItem(self->id_to_node, PyLong_FromLong(iter->first)), (PyObject*)temp_map));
            iter++;
        }
        ret = PyUnicode_Concat(ret, PyUnicode_FromFormat("}"));
        Py_DecRef((PyObject*)temp_map);
        break;
    }
    }
    return ret;
}

PyObject* GraphMap_copy(GraphMap* self, PyObject* args, PyObject* kwds) {
    GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
    temp_map->id_to_node = self->id_to_node;
    temp_map->node_to_id = self->node_to_id;
    switch (self->type) {
    case Msf: {
        temp_map->type = Msf;
        std::map<std::string, float>* map_pointer = new std::map<std::string, float>;
        *map_pointer = *(std::map<std::string, float>*)(self->pointer);
        temp_map->pointer = map_pointer;
        temp_map->flag = 1;
        return (PyObject*)temp_map;
    }
    case Mif: {
        temp_map->type = Mif;
        std::map<int, float>* map_pointer = new std::map<int, float>;
        *map_pointer = *(std::map<int, float>*)(self->pointer);
        temp_map->pointer = map_pointer;
        temp_map->flag = 1;
        return (PyObject*)temp_map;
    }
    case Mii: {
        temp_map->type = Mii;
        std::map<int, int>* map_pointer = new std::map<int, int>;
        *map_pointer = *(std::map<int, int>*)(self->pointer);
        temp_map->pointer = map_pointer;
        temp_map->flag = 1;
        return (PyObject*)temp_map;
    }
    case MiMsf: {
        temp_map->type = MiMsf;
        std::unordered_map<int, std::map<std::string, float>>* map_pointer = new std::unordered_map<int, std::map<std::string, float>>;
        *map_pointer = *(std::unordered_map<int, std::map<std::string, float>>*)(self->pointer);
        temp_map->pointer = map_pointer;
        temp_map->flag = 1;
        return (PyObject*)temp_map;
    }
    case MiMiMsf: {
        temp_map->type = MiMiMsf;
        std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>* map_pointer = new std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>;
        *map_pointer = *(std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>*)(self->pointer);
        temp_map->pointer = map_pointer;
        temp_map->flag = 1;
        return (PyObject*)temp_map;
    }
    }
}

PyObject* GraphMap_get(GraphMap* self, PyObject* args, PyObject* kwds) {
    PyObject* pykey, * default_val;
    PyArg_Parse(args, "(OO)", &pykey, &default_val);
    PyObject* ret = _GraphMap_getitem(self, pykey, default_val);
    return ret;
}

PyObject* GraphMap_keys(GraphMap* self, PyObject* args, PyObject* kwds) {
    PyObject* ret = PyObject_GetIter((PyObject*)self);
    return ret;
}

PyObject* GraphMap_values(GraphMap* self) {
    PyObject* ret = PyObject_GetIter((PyObject*)self);
    ((GraphMapIter*)ret)->flag = 1;
    return ret;
}

PyObject* GraphMap_items(GraphMap* self) {
    PyObject* ret = PyObject_GetIter((PyObject*)self);
    ((GraphMapIter*)ret)->flag = 2;
    return ret;
}
