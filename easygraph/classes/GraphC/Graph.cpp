#include "Graph.h"

//以下是类的属性方法
PyObject* Graph_get_graph(Graph* self, void*) {
    return self->graph;
}

PyObject* Graph_get_nodes(Graph* self, void*) {
    GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
    temp_map->id_to_node = self->id_to_node;
    temp_map->node_to_id = self->node_to_id;
    temp_map->type = MiMsf;
    temp_map->pointer = &(self->node);
    return (PyObject*)temp_map;
}

PyObject* Graph_get_adj(Graph* self, void*) {
    GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
    temp_map->id_to_node = self->id_to_node;
    temp_map->node_to_id = self->node_to_id;
    temp_map->type = MiMiMsf;
    temp_map->pointer = &(self->adj);
    return (PyObject*)temp_map;
}

PyObject* Graph_get_edges(Graph* self, void*) {
    GraphEdges* temp_edges = (GraphEdges*)PyObject_CallFunctionObjArgs((PyObject*)&GraphEdgesType, nullptr);
    temp_edges->id_to_node = self->id_to_node;
    temp_edges->node_to_id = self->node_to_id;
    std::vector<Edge>& edges = temp_edges->edges;
    std::unordered_set<long long> seen;
    std::unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>& adj = self->adj;
    edge_tuple temp_record;
    Edge temp_edge;
    for (auto& each : adj) {
        int u = each.first;
        for (auto& each_ : each.second) {
            int v = each_.first;
            temp_record = { u,v };
            if (!seen.count(temp_record.val)) {
                seen.insert(temp_record.val);
                temp_record = { v,u };
                seen.insert(temp_record.val);
                temp_edge = { u,v,&(each_.second) };
                edges.push_back(temp_edge);
            }
        }
    }
    return (PyObject*)temp_edges;
}

PyObject* Graph_get_map(Graph* self, void*) {
    return Py_BuildValue("(OO)", self->id_to_node, self->node_to_id);
}

PyGetSetDef Graph_get_set[] = {
    {"graph", (getter)Graph_get_graph, nullptr,"", NULL },
    {"nodes", (getter)Graph_get_nodes, nullptr,"", NULL },
    {"adj", (getter)Graph_get_adj, nullptr,"", NULL },
    {"edges",(getter)Graph_get_edges, nullptr, "", NULL},
    {"map", (getter)Graph_get_map, nullptr, "", NULL},
    {NULL}  /* Sentinel */
};

//以下是类的方法
void _add_one_node(Graph* self, PyObject* one_node_for_adding, PyObject* node_attr, std::map<std::string, float>* c_node_attr) {
    int id;
    if (PyDict_Contains(self->node_to_id, one_node_for_adding)) {
        id = PyLong_AsLong(PyDict_GetItem(self->node_to_id, one_node_for_adding));
    }
    else {
        id = ++(self->id);
        PyDict_SetItem(self->id_to_node, PyLong_FromLong(id), one_node_for_adding);
        PyDict_SetItem(self->node_to_id, one_node_for_adding, PyLong_FromLong(id));
    }
    if (c_node_attr) {
        self->node[id] = *c_node_attr;
    }
    else {
        PyObject* py_key, * py_value;
        Py_ssize_t pos = 0;
        if (node_attr != nullptr) {
            while (PyDict_Next(node_attr, &pos, &py_key, &py_value)) {
                std::string key(PyUnicode_AsUTF8(py_key));
                float value = float(PyLong_AsLong(py_value));
                self->node[id][key] = value;
            }
        }
        else
            if (!self->node.count(id))
                self->node[id] = std::map<std::string, float>();
    }
}

PyObject* Graph_add_node(Graph* self, PyObject* args, PyObject* kwargs) {
    if(PyTuple_Size(args) != 1) {
        PyErr_Format(PyExc_TypeError, "add_node() takes only 1 positional argument.");
        return nullptr;
    }
    PyObject* one_node_for_adding = PyTuple_GetItem(args, 0);
    _add_one_node(self, one_node_for_adding, kwargs);
    return Py_BuildValue("");
}

PyObject* Graph_add_nodes(Graph* self, PyObject* args, PyObject* kwargs) {
    
    PyObject* nodes_for_adding = nullptr, * nodes_attr = nullptr;
    static char* kwlist[] = { (char*)"nodes_for_adding", (char*)"nodes_attr", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &nodes_for_adding, &nodes_attr))
        return nullptr;
    Py_ssize_t nodes_for_adding_len = PyList_Size(nodes_for_adding);
    if (nodes_attr != nullptr && nodes_for_adding_len != PyList_Size(nodes_attr)) {
        PyErr_SetString(PyExc_AssertionError, "Nodes and Attributes lists must have same length.");
        return nullptr;
    }
    PyObject* node, * attr;
    for (Py_ssize_t i = 0;i < nodes_for_adding_len;i++) {
        node = PyList_GetItem(nodes_for_adding, i);
        attr = nodes_attr ? PyList_GetItem(nodes_attr, i) : nullptr;
        _add_one_node(self, node, attr);
    }
    return Py_BuildValue("");
}

void _add_one_edge(Graph* self, PyObject* pu, PyObject* pv, PyObject* edge_attr, std::map<std::string, float>* c_edge_attr) {
    int u, v;
    if (!PyDict_Contains(self->node_to_id, pu)) {
        _add_one_node(self, pu, nullptr);
        u = self->id;
    }
    else
        u = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pu));
    if (!PyDict_Contains(self->node_to_id, pv)) {
        _add_one_node(self, pv, nullptr);
        v = self->id;
    }
    else
        v = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pv));
    if (c_edge_attr) {
        self->adj[u][v] = *c_edge_attr;
        self->adj[v][u] = *c_edge_attr;
    }
    else {
        PyObject* py_key, * py_value;
        Py_ssize_t pos = 0;
        if (edge_attr != nullptr) {
            while (PyDict_Next(edge_attr, &pos, &py_key, &py_value)) {
                std::string key(PyUnicode_AsUTF8(py_key));
                float value = float(PyLong_AsLong(py_value));
                self->adj[u][v][key] = self->adj[v][u][key] = value;
            }
        }
        else {
            if (!self->adj[u].count(v)) {
                self->adj[u][v] = std::map<std::string, float>();
            }
            if (!self->adj[v].count(u)) {
                self->adj[v][u] = std::map<std::string, float>();
            }
        }
    }
}

PyObject* Graph_add_edge(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* u = nullptr, * v = nullptr;
    if(PyTuple_Size(args) != 2) {
        PyErr_Format(PyExc_TypeError, "add_edge() takes only 2 positional arguments.");
        return nullptr;
    }
    PyArg_ParseTuple(args, "OO", &u, &v);
    _add_one_edge(self, u, v, kwargs);
    return Py_BuildValue("");
}

PyObject* Graph_add_weighted_edge(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* pu, * pv;
    float weight;
    static char* kwlist[] = { (char*)"u_of_edge", (char*)"v_of_edge", (char*)"weight", NULL };
    std::string key("weight");
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOf", kwlist, &pu, &pv, &weight))
        return nullptr;
    int u, v;
    if (!PyDict_Contains(self->node_to_id, pu)) {
        _add_one_node(self, pu, nullptr);
        u = self->id;
    }
    else
        u = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pu));
    if (!PyDict_Contains(self->node_to_id, pv)) {
        _add_one_node(self, pv, nullptr);
        v = self->id;
    }
    else
        v = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pv));
    self->adj[u][v][key] = self->adj[v][u][key] = weight;
    return Py_BuildValue("");
}

PyObject * Graph_add_edges(Graph * self, PyObject* args, PyObject* kwargs) {
    PyObject* edges_for_adding, * edges_attr = nullptr;
    static char* kwlist[] = { (char*)"edges_for_adding", (char*)"edges_attr", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &edges_for_adding, &edges_attr))
        return nullptr;
    Py_ssize_t edges_for_adding_len = PyList_Size(edges_for_adding);
    if (edges_attr != nullptr && edges_for_adding_len != PyList_Size(edges_attr)) {
        PyErr_SetString(PyExc_AssertionError, "Edges and Attributes lists must have same length.");
        return nullptr;
    }
    PyObject* edge, * attr;
    for (Py_ssize_t i = 0;i < edges_for_adding_len;i++) {
        edge = PyList_GetItem(edges_for_adding, i);
        attr = edges_attr ? PyList_GetItem(edges_attr, i) : nullptr;
        if (PyTuple_Size(edge) != 2) {
            PyErr_SetString(PyExc_AssertionError, "Edge tuple must be 2 - tuple.");
            return nullptr;
        }
        else {
            PyObject* u = nullptr, * v = nullptr;
            PyArg_ParseTuple(edge, "OO", &u, &v);
            _add_one_edge(self, u, v, attr);
        }
    }
    return Py_BuildValue("");
}

PyObject* Graph_add_edges_from_file(Graph* self, PyObject* args, PyObject* kwargs) {
    std::ios::sync_with_stdio(0);
    char* file_path;
    PyObject* weighted = Py_False;
    static char* kwlist[] = { (char*)"file", (char*)"weighted", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|O", kwlist, &file_path, &weighted))
        return nullptr;
    std::ifstream in;
    in.open(file_path, std::ios::in);
    in.imbue(std::locale(std::locale(), new commactype));
    std::string data, key("weight");
    std::string su, sv;
    float weight;
    while (in >> su >> sv) {
        PyObject* pu = PyUnicode_FromString(su.c_str()), * pv = PyUnicode_FromString(sv.c_str());
        int u, v;
        if (!PyDict_Contains(self->node_to_id, pu)) {
            _add_one_node(self, pu, nullptr);
            u = self->id;
        }
        else
            u = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pu));
        if (!PyDict_Contains(self->node_to_id, pv)) {
            _add_one_node(self, pv, nullptr);
            v = self->id;
        }
        else
            v = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pv));
        if (PyObject_IsTrue(weighted)) {
            in >> weight;
            self->adj[u][v][key] = self->adj[v][u][key] = weight;
        }
        else {
            if (!self->adj[u].count(v)) {
                self->adj[u][v] = std::map<std::string, float>();
            }
            if (!self->adj[v].count(u)) {
                self->adj[v][u] = std::map<std::string, float>();
            }
        }
    }
    in.close();
    return Py_BuildValue("");
}

PyObject* Graph_degree(Graph* self, PyObject* args, PyObject* kwargs) {
    char* weight_key_cstr = (char*)"weight";
    static char* kwlist[] = { (char*)"weight", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|s", kwlist, &weight_key_cstr))
        return nullptr;
    std::string weight_key(weight_key_cstr);
    GraphEdges* temp_edges = (GraphEdges*)PyObject_GetAttr((PyObject*)self, PyUnicode_FromString("edges"));
    GraphMap* temp_degree = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
    temp_degree->type = Mif;
    std::unordered_map<int, float>* degree_pointer = new std::unordered_map<int, float>;
    std::unordered_map<int, float>& degree = *degree_pointer;
    for (auto& each : self->node) {
        degree[each.first] = 0;
    }
    for (auto& edge : temp_edges->edges) {
        float weight = (*edge.weight).count(weight_key) ? (*edge.weight)[weight_key] : 0;
        degree[edge.u] += weight + !weight;
        degree[edge.v] += weight + !weight;
    }
    temp_degree->pointer = degree_pointer;
    temp_degree->flag = 1;
    temp_degree->id_to_node = self->id_to_node;
    temp_degree->node_to_id = self->node_to_id;
    return (PyObject*)temp_degree;
}

PyObject* Graph_size(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* weight = Py_None;
    static char* kwlist[] = { (char*)"weight", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &weight))
        return nullptr;
    GraphMap* temp_degree = (GraphMap*)PyObject_CallMethod((PyObject*)self, "degree", "(O)", weight);
    std::unordered_map<int, float>* degree_pointer = (std::unordered_map<int, float>*)temp_degree->pointer;
    float result = 0;
    for (auto& each : *degree_pointer) {
        result += each.second;
    }
    Py_DecRef((PyObject*)temp_degree);
    return weight == Py_None? Py_BuildValue("i", int(result) / 2): Py_BuildValue("f", result / 2);
        
}

PyObject* Graph_neighbors(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* node;
    static char* kwlist[] = { (char*)"node", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &node))
        return nullptr;
    PyObject* temp_adj = PyObject_GetAttr((PyObject*)self, PyUnicode_FromString("adj"));
    PyObject* obj = PyObject_GetItem(temp_adj, node);
    auto& adj = self->adj;
    if (PyDict_Contains(self->node_to_id, node)) {
        int id = PyLong_AsLong(PyDict_GetItem(self->node_to_id, node));
        GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
        temp_map->node_to_id = self->node_to_id;
        temp_map->id_to_node = self->id_to_node;
        temp_map->type = MiMsf;
        temp_map->pointer = &adj[id];
        return GraphMap_iter(temp_map);
    }else {
        PyErr_Format(PyExc_KeyError, "No node %R", node);
        return nullptr;
    }
}

void _remove_one_node(Graph* self, PyObject* node_to_remove) {
    PyObject* pid = PyDict_GetItem(self->node_to_id, node_to_remove);
    int id = PyLong_AsLong(pid);
    PyDict_DelItem(self->node_to_id, node_to_remove);
    PyDict_DelItem(self->id_to_node, pid);
    self->node.erase(id);
    auto& temp_adj = self->adj;
    auto& temp_neighbors = temp_adj[id];
    for (auto& temp_neighbor_pair : temp_neighbors) {
        temp_adj[temp_neighbor_pair.first].erase(id);
    }
    temp_adj.erase(id);
}

PyObject* Graph_remove_node(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* node_to_remove;
    static char* kwlist[] = { (char*)"node_to_remove", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &node_to_remove)) {
        return nullptr;
    }
    _remove_one_node(self, node_to_remove);
    return Py_BuildValue("");
} 

PyObject* Graph_remove_nodes(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* nodes_to_remove;
    static char* kwlist[] = { (char*)"nodes_to_remove", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &nodes_to_remove))
        return nullptr;
    if (!PyList_Check(nodes_to_remove)) {
        PyErr_Format(PyExc_TypeError, "Error: The type of the parameter should be list.");
        return nullptr;
    }
    Py_ssize_t nodes_len = PyList_Size(nodes_to_remove);
    for (Py_ssize_t i = 0;i < nodes_len;i++) {
        PyObject* node = PyList_GetItem(nodes_to_remove, i);
        if (!PyDict_Contains(self->node_to_id, node)) {
            PyErr_Format(PyExc_AssertionError, "Remove Error: No node %R in graph.", node);
            return nullptr;
        }
    }
    for (Py_ssize_t i = 0;i < nodes_len;i++)
        _remove_one_node(self, PyList_GetItem(nodes_to_remove, i));
    return Py_BuildValue("");
}

void _remove_one_edge(Graph* self, int u, int v) {
    auto& temp_adj = self->adj;
    temp_adj[u].erase(v);
    if (u != v)
        temp_adj[v].erase(u);
}

PyObject* Graph_remove_edge(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* pu , * pv;
    static char* kwlist[] = { (char*)"u", (char*)"v", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &pu, &pv))
        return nullptr;
    int u = -1, v = -1;
    if (PyDict_Contains(self->node_to_id, pu) && PyDict_Contains(self->node_to_id, pv)) {
        u = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pu));
        v = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pv));
        if (self->adj.count(u) && self->adj[u].count(v)) {
            _remove_one_edge(self, u, v);
            return Py_BuildValue("");
        }
    }
    else {
        PyErr_Format(PyExc_KeyError, "No edge %R-%R in graph.", pu, pv);
        return nullptr;
    }
}

PyObject* Graph_remove_edges(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* edges_to_remove;
    static char* kwlist[] = { (char*)"edges_to_remove", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &edges_to_remove))
        return nullptr;
    if (!PyList_Check(edges_to_remove)) {
        PyErr_Format(PyExc_TypeError, "Remove Error: The type of the parameter should be list.");
        return nullptr;
    }
    Py_ssize_t edges_len = PyList_Size(edges_to_remove);
    auto& temp_adj = self->adj;
    struct edge {
        int u, v;
    };
    std::vector<edge> edges;
    for (Py_ssize_t i = 0;i < edges_len;i++) {
        PyObject* edge_to_remove = PyList_GetItem(edges_to_remove, i);
        PyObject* pu = nullptr, * pv = nullptr;
        PyArg_ParseTuple(edge_to_remove, "OO", &pu, &pv);
        int u = -1, v = -1;
        if (PyDict_Contains(self->node_to_id, pu) && PyDict_Contains(self->node_to_id, pv)) {
            u = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pu));
            v = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pv));
            if (!(self->adj.count(u) && self->adj[u].count(v))) {
                PyErr_Format(PyExc_KeyError, "No edge %R-%R in graph.", pu, pv);
                return nullptr;
            }
            else {
                edges.push_back({ u,v });
            }
        }
        else {
            PyErr_Format(PyExc_KeyError, "No edge %R-%R in graph.", pu, pv);
            return nullptr;
        }
    }
    for (auto& each:edges) {
        _remove_one_edge(self, each.u, each.v);
    }
    return Py_BuildValue("");
}

PyObject* Graph_has_node(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* node;
    static char* kwlist[] = { (char*)"node", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &node))
        return nullptr;
    return PyDict_Contains(self->node_to_id, node) ? Py_True : Py_False;
}

PyObject* Graph_has_edge(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* pu, * pv;
    static char* kwlist[] = { (char*)"u", (char*)"v", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &pu, &pv))
        return nullptr;
    int u = -1, v = -1;
    if (PyDict_Contains(self->node_to_id, pu) && PyDict_Contains(self->node_to_id, pv)) {
        u = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pu));
        v = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pv));
        if (self->adj.count(u) && self->adj[u].count(v)) {
            return Py_True;
        }
    }
    return Py_False;
}

PyObject* Graph_number_of_nodes(Graph* self, PyObject* args, PyObject* kwargs) {
    return Py_BuildValue("i", self->node.size());
}

PyObject* Graph_number_of_edges(Graph* self, PyObject* args, PyObject* kwargs) {
    return PyObject_CallMethod((PyObject*)self, "size", "()");
}

PyObject* Graph_is_directed(Graph* self, PyObject* args, PyObject* kwargs) {
    return Py_False;
}

PyMethodDef GraphMethods[] = {
    {"add_node", (PyCFunction)Graph_add_node, METH_VARARGS | METH_KEYWORDS, "" },
    {"add_nodes", (PyCFunction)Graph_add_nodes, METH_VARARGS | METH_KEYWORDS, "" },
    {"add_edge", (PyCFunction)Graph_add_edge, METH_VARARGS | METH_KEYWORDS, "" },
    {"add_edges", (PyCFunction)Graph_add_edges, METH_VARARGS | METH_KEYWORDS, ""},
    {"add_weighted_edge", (PyCFunction)Graph_add_weighted_edge, METH_VARARGS | METH_KEYWORDS, ""},
    {"add_edges_from_file", (PyCFunction)Graph_add_edges_from_file, METH_VARARGS | METH_KEYWORDS, ""},
    {"degree", (PyCFunction)Graph_degree, METH_VARARGS | METH_KEYWORDS, ""},
    {"size", (PyCFunction)Graph_size, METH_VARARGS | METH_KEYWORDS, ""},
    {"neighbors", (PyCFunction)Graph_neighbors, METH_VARARGS | METH_KEYWORDS, ""},
    {"all_neighbors", (PyCFunction)Graph_neighbors, METH_VARARGS | METH_KEYWORDS, ""},
    {"remove_node", (PyCFunction)Graph_remove_node, METH_VARARGS | METH_KEYWORDS, ""},
    {"remove_nodes", (PyCFunction)Graph_remove_nodes, METH_VARARGS | METH_KEYWORDS, ""},
    {"remove_edge", (PyCFunction)Graph_remove_edge, METH_VARARGS | METH_KEYWORDS, ""},
    {"remove_edges", (PyCFunction)Graph_remove_edges, METH_VARARGS | METH_KEYWORDS, ""},
    {"has_node", (PyCFunction)Graph_has_node, METH_VARARGS | METH_KEYWORDS, ""},
    {"has_edge", (PyCFunction)Graph_has_edge, METH_VARARGS | METH_KEYWORDS, ""},
    {"number_of_nodes", (PyCFunction)Graph_number_of_nodes, METH_VARARGS | METH_KEYWORDS, ""},
    {"number_of_edges", (PyCFunction)Graph_number_of_edges, METH_VARARGS | METH_KEYWORDS, ""},
    {"is_directed", (PyCFunction)Graph_is_directed, METH_VARARGS | METH_KEYWORDS, ""},
    {"copy", (PyCFunction)Graph_copy, METH_VARARGS | METH_KEYWORDS, ""},
    {"nodes_subgraph", (PyCFunction)Graph_nodes_subgraph, METH_VARARGS | METH_KEYWORDS, ""},
    {"ego_subgraph", (PyCFunction)Graph_ego_subgraph, METH_VARARGS | METH_KEYWORDS, ""},
    {"to_index_node_graph", (PyCFunction)Graph_to_index_node_graph, METH_VARARGS | METH_KEYWORDS, ""},
    {NULL}
};

//以下是作为sequence的方法
Py_ssize_t Graph_len(Graph* self) {
    return self->node.size();
}

int Graph_contains(Graph* self, PyObject* node) {
    return PyDict_Contains(self->node_to_id, node);
}

PySequenceMethods Graph_sequence_methods = {
    (lenfunc)Graph_len,                    /* sq_length */
    nullptr,                               /* sq_concat */
    nullptr,                               /* sq_repeat */
    nullptr,                               /* sq_item */
    nullptr,                               /* was_sq_slice; */
    nullptr,                               /* sq_ass_item; */
    nullptr,                               /* was_sq_ass_slice */
    (objobjproc)Graph_contains,            /* sq_contains */
    nullptr,                               /* sq_inplace_concat */
    nullptr                                /* sq_inplace_repeat */
};

//以下是作为mapping的方法
PyObject* Graph_getitem(Graph* self, PyObject* pykey) {
    PyObject* pkey = PyTuple_GetItem(pykey, 0);
    if (PyDict_Contains(self->node_to_id, pkey)) {
        int key = PyLong_AsLong(PyDict_GetItem(self->node_to_id, pkey));
        GraphMap* temp_map = (GraphMap*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapType, nullptr);
        temp_map->pointer = &(self->adj[key]);
        temp_map->type = MiMsf;
        temp_map->node_to_id = self->node_to_id;
        temp_map->id_to_node = self->id_to_node;
        return (PyObject*)temp_map;
    }
    else {
        PyErr_SetString(PyExc_KeyError, "key not found");
        return nullptr;
    }
}

PyMappingMethods Graph_mapping_methods = {
    nullptr,                               /* mp_length */
    (binaryfunc)Graph_getitem,             /* mp_subscript */
    nullptr,                               /* mp_ass_subscript */
};

//以下是类的内置方法
PyObject* Graph_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    Graph* self;
    self = (Graph*)type->tp_alloc(type, 0);
    new (self)Graph;
    self->graph = kwds ? kwds : PyDict_New();
    self->id = -1;
    Py_IncRef(kwds);
    self->id_to_node = PyDict_New();
    self->node_to_id = PyDict_New();
    return (PyObject*)self;
}

void* Graph_dealloc(PyObject* obj) {
    Graph* self = (Graph*)obj;
    self->node.~unordered_map<int, std::map<std::string, float>>();
    self->adj.~unordered_map<int, std::unordered_map<int, std::map<std::string, float>>>();
    Py_TYPE(self->id_to_node)->tp_free(self->id_to_node);
    Py_TYPE(self->node_to_id)->tp_free(self->node_to_id);
    Py_TYPE(obj)->tp_free(obj);
    return nullptr;
}

PyObject* Graph_iter(Graph* self) {
    GraphMapIter* temp_iter = (GraphMapIter*)PyObject_CallFunctionObjArgs((PyObject*)&GraphMapIterType, nullptr);
    temp_iter->id_to_node = self->id_to_node;
    temp_iter->node_to_id = self->node_to_id;
    temp_iter->type = MiMsf;
    temp_iter->MiMsf_iter = self->node.begin();
    temp_iter->MiMsf_end = self->node.end();
    return (PyObject*)temp_iter;
}

//Type对象定义
PyTypeObject GraphType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "cpp_easygraph.Graph",                             /* tp_name */
    sizeof(Graph),                                     /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)Graph_dealloc,                         /* tp_dealloc */
    0,                                                 /* tp_vectorcall_offset */
    nullptr,                                           /* tp_getattr */
    nullptr,                                           /* tp_setattr */
    nullptr,                                           /* tp_as_async */
    nullptr,                                           /* tp_repr */
    nullptr,                                           /* tp_as_number */
    &Graph_sequence_methods,                           /* tp_as_sequence */
    &Graph_mapping_methods,                            /* tp_as_mapping */
    nullptr,                                           /* tp_hash  */
    nullptr,                                           /* tp_call */
    nullptr,                                           /* tp_str */
    nullptr,                                           /* tp_getattro */
    nullptr,                                           /* tp_setattro */
    nullptr,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /* tp_flags */
    "easygraph writed with c++.",                      /* tp_doc */
    nullptr,                                           /* tp_traverse */
    nullptr,                                           /* tp_clear */
    nullptr,                                           /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    (getiterfunc)Graph_iter,                           /* tp_iter */
    nullptr,                                           /* tp_iternext */
    GraphMethods,                                      /* tp_methods */
    nullptr,                                           /* tp_members */
    Graph_get_set,                                     /* tp_getset */
    nullptr,                                           /* tp_base */
    nullptr,                                           /* tp_dict */
    nullptr,                                           /* tp_descr_get */
    nullptr,                                           /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    nullptr,                                           /* tp_init */
    nullptr,                                           /* tp_alloc */
    Graph_new,                                         /* tp_new */
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

PyObject* Graph_copy(Graph* self, PyObject* args, PyObject* kwargs) {
    Graph* temp_graph = (Graph*)PyObject_CallFunctionObjArgs((PyObject*)&GraphType, nullptr);
    temp_graph->graph = PyDict_Copy(self->graph);
    temp_graph->node = self->node;
    temp_graph->adj = self->adj;
    temp_graph->id_to_node = PyDict_Copy(self->id_to_node);
    temp_graph->node_to_id = PyDict_Copy(self->node_to_id);
    return (PyObject*)temp_graph;
}

PyObject* Graph_nodes_subgraph(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* from_nodes;
    static char* kwlist[] = { (char*)"from_nodes", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &from_nodes))
        return nullptr;
    Graph* temp_graph = (Graph*)PyObject_CallFunctionObjArgs((PyObject*)&GraphType, nullptr);
    temp_graph->graph = PyDict_Copy(self->graph);
    Py_ssize_t len = PyList_Size(from_nodes);
    std::unordered_set<int> _from_nodes;
    for (Py_ssize_t i = 0;i < len;i++) {
        _from_nodes.insert(PyLong_AsLong(PyDict_GetItem(self->node_to_id, PyList_GetItem(from_nodes, i))));
    }
    auto& nodes = self->node;
    for (const int& each : _from_nodes) {
        if (nodes.count(each))
            _add_one_node(temp_graph, PyDict_GetItem(self->id_to_node, PyLong_FromLong(each)), nullptr, &(nodes[each]));
        else
            continue;
        for (auto& each_pair : self->adj[each]) {
            if (_from_nodes.count(each_pair.first))
                _add_one_edge(temp_graph, PyDict_GetItem(self->id_to_node, PyLong_FromLong(each)), PyDict_GetItem(self->id_to_node, PyLong_FromLong(each_pair.first)), nullptr, &(each_pair.second));
        }
    }
    return (PyObject*)temp_graph;
}

PyObject* Graph_ego_subgraph(Graph* self, PyObject* args, PyObject* kwargs) {
    PyObject* center;
    static char* kwlist[] = { (char*)"center", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &center))
        return nullptr;
    Graph* temp_graph = (Graph*)PyObject_CallFunctionObjArgs((PyObject*)&GraphType, nullptr);
    temp_graph->graph = PyDict_Copy(self->graph);
    PyObject* neighbors_of_center = PySequence_List(PyObject_CallMethod((PyObject*)self, "neighbors", "(O)", center));
    PyList_Append(neighbors_of_center, center);
    return PyObject_CallMethod((PyObject*)self, "nodes_subgraph", "(O)", neighbors_of_center);
}

PyObject* Graph_to_index_node_graph(Graph* self, PyObject* args, PyObject* kwargs) {
    int begin_index = 0;
    static char* kwlist[] = { (char*)"begin_index", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &begin_index))
        return nullptr;
    Graph* G = (Graph*)PyObject_CallFunctionObjArgs((PyObject*)&GraphType, nullptr);
    G->graph = PyDict_Copy(self->graph);
    auto& G_node = G->node;
    auto& G_adj = G->adj;
    PyObject* index_of_node = PyDict_New(), * node_of_index = PyDict_New();
    begin_index--;
    for (auto& each_node : self->node) {
        ++begin_index;
        PyObject* pid = PyLong_FromLong(begin_index);
        PyObject* node = PyDict_GetItem(self->id_to_node, PyLong_FromLong(each_node.first));
        G_node.insert(make_pair(begin_index, each_node.second));
        PyDict_SetItem(G->node_to_id, pid, pid);
        PyDict_SetItem(G->id_to_node, pid, pid);
        PyDict_SetItem(index_of_node, node, pid);
        PyDict_SetItem(node_of_index, pid, node);
    }
    G->id = begin_index;
    for (auto& each_edge : self->adj) {
        for (auto& each_edge_terminal : self->adj[each_edge.first]) {
            int u = PyLong_AsLong(PyDict_GetItem(index_of_node, PyDict_GetItem(self->id_to_node, PyLong_FromLong(each_edge.first))));
            int v = PyLong_AsLong(PyDict_GetItem(index_of_node, PyDict_GetItem(self->id_to_node, PyLong_FromLong(each_edge_terminal.first))));
            G_adj[u][v] = G_adj[v][u] = each_edge_terminal.second;
        }
    }
    return Py_BuildValue("(OOO)", G, index_of_node, node_of_index);
}
