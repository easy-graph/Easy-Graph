#include "ModuleMethods.h"

PyMethodDef ModuleMethods[] = {  //method list
    {"cpp_effective_size", (PyCFunction)effective_size, METH_VARARGS | METH_KEYWORDS, "" },
    {"cpp_constraint", (PyCFunction)constraint, METH_VARARGS | METH_KEYWORDS, "" },
    {"cpp_hierarchy", (PyCFunction)hierarchy, METH_VARARGS | METH_KEYWORDS, "" },
    {NULL}
};

//module对象定义
PyModuleDef Graphmodule = {
    PyModuleDef_HEAD_INIT,
    "cpp_easygraph", //module name
    "easygraph writed with C++", //docstring
    -1, //Multiple interpreter
    ModuleMethods }; //method list

PyObject* initModule(void) {
    PyObject* m;
    if (PyType_Ready(&GraphType) < 0)
        return NULL;
  
    if (PyType_Ready(&GraphMapIterType) < 0)
        return NULL;

    if (PyType_Ready(&GraphMapType) < 0)
        return NULL;

    if (PyType_Ready(&GraphEdgesType) < 0)
        return NULL;

    if (PyType_Ready(&GraphEdgesIterType) < 0)
        return NULL;

    if (PyType_Ready(&GraphEdgeType) < 0)
        return NULL;

    m = PyModule_Create(&Graphmodule);
    if (m == NULL)
        return NULL;

    Py_IncRef((PyObject*)&GraphType);
    Py_IncRef((PyObject*)&GraphMapIterType);
    Py_IncRef((PyObject*)&GraphMapType);
    Py_IncRef((PyObject*)&GraphEdgesType);
    Py_IncRef((PyObject*)&GraphEdgesIterType);
    Py_IncRef((PyObject*)&GraphEdgeType);
    if (PyModule_AddObject(m, "Graph", (PyObject*)&GraphType) < 0) {
        Py_DecRef((PyObject*)&GraphType);
        Py_DecRef(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "GraphMapIter", (PyObject*)&GraphMapIterType) < 0) {
        Py_DecRef((PyObject*)&GraphMapIterType);
        Py_DecRef(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "GraphMap", (PyObject*)&GraphMapType) < 0) {
        Py_DecRef((PyObject*)&GraphMapType);
        Py_DecRef(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "GraphEdges", (PyObject*)&GraphEdgesType) < 0) {
        Py_DecRef((PyObject*)&GraphEdgesType);
        Py_DecRef(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "GraphEdgesIter", (PyObject*)&GraphEdgesIterType) < 0) {
        Py_DecRef((PyObject*)&GraphEdgesIterType);
        Py_DecRef(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "GraphEdge", (PyObject*)&GraphEdgeType) < 0) {
        Py_DecRef((PyObject*)&GraphEdgeType);
        Py_DecRef(m);
        return NULL;
    }

    return m;
}

PyMODINIT_FUNC
PyInit_cpp_easygraph(void)
{
    return initModule();
}