#include "directed_graph.h"

#include "../common/utils.h"

DiGraph::DiGraph() : Graph() {

}

py::object DiGraph__init__(py::args args, py::kwargs kwargs) {
    py::object self = args[0];
    self.attr("__init__")();
    DiGraph& self_ = self.cast<DiGraph&>();
    py::dict graph_attr = kwargs;
    self_.graph.attr("update")(graph_attr);
    self_.nodes_cache = py::dict();
    self_.adj_cache = py::dict();
    return py::none();
}

py::object DiGraph_out_degree(py::object self, py::object weight) {
    py::dict degree = py::dict();
    py::list edges = self.attr("edges").cast<py::list>();
    py::object u, v;
    py::dict d;
    for (int i = 0; i < py::len(edges); i++) {
        py::tuple edge = edges[i].cast<py::tuple>();
        u = edge[0];
        v = edge[1];
        d = edge[2].cast<py::dict>();
        if (degree.contains(u)) {
            degree[u] = py::object(degree[u]) + d.attr("get")(weight, 1);
        } else {
            degree[u] = d.attr("get")(weight, 1);
        }
    }
    py::list nodes = py::list(self.attr("nodes"));
    for (int i = 0; i < py::len(nodes); i++) {
        py::object node = nodes[i];
        if (!degree.contains(node)) {
            degree[node] = 0;
        }
    }
    return degree;
}

py::object DiGraph_in_degree(py::object self, py::object weight) {
    py::dict degree = py::dict();
    py::list edges = self.attr("edges").cast<py::list>();
    py::object u, v;
    py::dict d;
    for (int i = 0; i < py::len(edges); i++) {
        py::tuple edge = edges[i].cast<py::tuple>();
        u = edge[0];
        v = edge[1];
        d = edge[2].cast<py::dict>();
        if (degree.contains(v)) {
            degree[v] = py::object(degree[v]) + d.attr("get")(weight, 1);
        } else {
            degree[v] = d.attr("get")(weight, 1);
        }
    }
    py::list nodes = py::list(self.attr("nodes"));
    for (int i = 0; i < py::len(nodes); i++) {
        py::object node = nodes[i];
        if (!degree.contains(node)) {
            degree[node] = 0;
        }
    }
    return degree;
}

py::object DiGraph_degree(py::object self, py::object weight) {
    py::dict degree = py::dict();
    py::dict out_degree = self.attr("out_degree")(weight).cast<py::dict>();
    py::dict in_degree = self.attr("in_degree")(weight).cast<py::dict>();
    py::list nodes = py::list(self.attr("nodes"));
    for (int i = 0; i < py::len(nodes); i++) {
        py::object u = nodes[i];
        degree[u] = out_degree[u] + in_degree[u];
    }
    return degree;
}

py::object DiGraph_size(py::object self, py::object weight) {
    py::dict out_degree = self.attr("out_degree")(weight).cast<py::dict>();
    py::object s = py_sum(out_degree.attr("values")());
    return (weight.is_none()) ? py::int_(s) : s;
}

py::object DiGraph_number_of_edges(py::object self, py::object u, py::object v) {
    if (u.is_none()) {
        return self.attr("size")();
    }
    Graph& G = self.cast<Graph&>();
    node_t u_id = G.node_to_id[u].cast<node_t>();
    node_t v_id = G.node_to_id.attr("get")(v, -1).cast<node_t>();
    return py::cast(int(v_id != -1 && G.adj[u_id].count(v_id)));
}

py::object DiGraph_neighbors(py::object self, py::object node) {
    Graph& self_ = self.cast<Graph&>();
    if (self_.node_to_id.contains(node)) {
        return self.attr("adj")[node].attr("__iter__")();
    } else {
        PyErr_Format(PyExc_KeyError, "No node %R", node.ptr());
        return py::none();
    }
}

py::object DiGraph_predecessors(py::object self, py::object node) {
    DiGraph& self_ = self.cast<DiGraph&>();
    adj_dict_factory& pred = self_.pred;
    node_t node_id = self_.node_to_id[node].cast<node_t>();
    if (pred.find(node_id) != pred.end()) {
        adj_attr_dict_factory node_pred_dict = pred[node_id];
        py::dict node_pred = py::dict();
        for (adj_attr_dict_factory::iterator i = node_pred_dict.begin();
             i != node_pred_dict.end(); i++) {
            edge_attr_dict_factory edge_attr_dict = i->second;
            node_pred[self_.id_to_node[py::cast(i->first)]] = attr_to_dict(edge_attr_dict);
        }
        return node_pred.attr("__iter__")();
    } else {
        PyErr_Format(PyExc_KeyError, "No node %R", node.ptr());
        return py::none();
    }
}

node_t DiGraph_add_one_node(DiGraph& self, py::object one_node_for_adding,
                            py::object node_attr = py::dict()) {
    node_t id;
    if (self.node_to_id.contains(one_node_for_adding)) {
        id = self.node_to_id[one_node_for_adding].cast<node_t>();
    } else {
        id = ++(self.id);
        self.id_to_node[py::cast(id)] = one_node_for_adding;
        self.node_to_id[one_node_for_adding] = id;
    }
    py::list items = py::list(node_attr.attr("items")());
    self.node[id] = node_attr_dict_factory();
    self.adj[id] = adj_attr_dict_factory();
    self.pred[id] = adj_attr_dict_factory();

    for (int i = 0; i < len(items); i++) {
        py::tuple kv = items[i].cast<py::tuple>();
        py::object pkey = kv[0];
        std::string weight_key = weight_to_string(pkey);
        weight_t value = kv[1].cast<weight_t>();
        self.node[id].insert(std::make_pair(weight_key, value));
    }
    return id;
}

py::object DiGraph_add_node(py::args args, py::kwargs kwargs) {
    DiGraph& self = args[0].cast<DiGraph&>();
    self.dirty_nodes = true;
    self.dirty_adj = true;
    py::object one_node_for_adding = args[1];
    py::dict node_attr = kwargs;
    DiGraph_add_one_node(self, one_node_for_adding, node_attr);
    return py::none();
}

py::object DiGraph_add_nodes(DiGraph& self, py::list nodes_for_adding, py::list nodes_attr) {
    self.dirty_nodes = true;
    self.dirty_adj = true;
    if (py::len(nodes_attr) != 0) {
        if (py::len(nodes_for_adding) != py::len(nodes_attr)) {
            PyErr_Format(PyExc_AssertionError, "Nodes and Attributes lists must have same length.");
            return py::none();
        }
    }
    for (int i = 0; i < py::len(nodes_for_adding); i++) {
        py::object one_node_for_adding = nodes_for_adding[i];
        py::dict node_attr;
        if (py::len(nodes_attr)) {
            node_attr = nodes_attr[i].cast<py::dict>();
        } else {
            node_attr = py::dict();
        }
        DiGraph_add_one_node(self, one_node_for_adding, node_attr);
    }
    return py::none();
}

py::object DiGraph_add_nodes_from(py::args args, py::kwargs kwargs) {
    DiGraph& self = args[0].cast<DiGraph&>();
    self.dirty_nodes = true;
    self.dirty_adj = true;
    py::list nodes_for_adding = py::list(args[1]);
    for (int i = 0; i < py::len(nodes_for_adding); i++) {
        bool newnode;
        py::dict attr = kwargs;
        py::dict newdict, ndict;
        py::object n = nodes_for_adding[i];
        try {
            newnode = !self.node_to_id.contains(n);
            newdict = attr;
        } catch (const py::error_already_set&) {
            PyObject *type, *value, *traceback;
            PyErr_Fetch(&type, &value, &traceback);
            if (PyErr_GivenExceptionMatches(PyExc_TypeError, type)) {
                py::tuple n_pair = n.cast<py::tuple>();
                n = n_pair[0];
                ndict = n_pair[1].cast<py::dict>();
                newnode = !self.node_to_id.contains(n);
                newdict = attr.attr("copy")();
                newdict.attr("update")(ndict);
            } else {
                PyErr_Restore(type, value, traceback);
                return py::none();
            }
        }
        if (newnode) {
            if (n.is_none()) {
                PyErr_Format(PyExc_ValueError, "None cannot be a node");
                return py::none();
            }
            DiGraph_add_one_node(self, n);
        }
        node_t id = self.node_to_id[n].cast<node_t>();
        py::list items = py::list(newdict.attr("items")());
        for (int i = 0; i < len(items); i++) {
            py::tuple kv = items[i].cast<py::tuple>();
            py::object pkey = kv[0];
            std::string weight_key = weight_to_string(pkey);
            weight_t value = kv[1].cast<weight_t>();
            self.node[id].insert(std::make_pair(weight_key, value));
        }
    }
    return py::none();
}

py::object DiGraph_remove_node(DiGraph& self, py::object node_to_remove) {
    self.dirty_nodes = true;
    self.dirty_adj = true;
    if (!self.node_to_id.contains(node_to_remove)) {
        PyErr_Format(PyExc_KeyError, "No node %R in graph.", node_to_remove.ptr());
        return py::none();
    }
    node_t node_to_remove_id = self.node_to_id[node_to_remove].cast<node_t>();
    adj_attr_dict_factory succs = self.adj[node_to_remove_id];
    adj_attr_dict_factory preds = self.pred[node_to_remove_id];
    self.node.erase(node_to_remove_id);
    for (adj_attr_dict_factory::iterator i = succs.begin(); i != succs.end(); i++) {
        self.pred[i->first].erase(node_to_remove_id);
    }
    for (adj_attr_dict_factory::iterator i = preds.begin(); i != preds.end(); i++) {
        self.adj[i->first].erase(node_to_remove_id);
    }
    self.adj.erase(node_to_remove_id);
    self.pred.erase(node_to_remove_id);
    self.node_to_id.attr("pop")(node_to_remove);
    self.id_to_node.attr("pop")(node_to_remove_id);
    return py::none();
}

py::object DiGraph_remove_nodes(py::object self, py::list nodes_to_remove) {
    DiGraph& self_ = self.cast<DiGraph&>();
    self_.dirty_nodes = true;
    self_.dirty_adj = true;
    for (int i = 0; i < py::len(nodes_to_remove); i++) {
        py::object node_to_remove = nodes_to_remove[i];
        if (!self_.node_to_id.contains(node_to_remove)) {
            PyErr_Format(PyExc_KeyError, "No node %R in graph.", node_to_remove.ptr());
            return py::none();
        }
    }
    for (int i = 0; i < py::len(nodes_to_remove); i++) {
        py::object node_to_remove = nodes_to_remove[i];
        self.attr("remove_node")(node_to_remove);
    }
    return py::none();
}

void DiGraph_add_one_edge(DiGraph& self, py::object u_of_edge,
                          py::object v_of_edge, py::object edge_attr) {
    node_t u, v;
    if (!self.node_to_id.contains(u_of_edge)) {
        u = DiGraph_add_one_node(self, u_of_edge);
    } else {
        u = self.node_to_id[u_of_edge].cast<node_t>();
    }
    if (!self.node_to_id.contains(v_of_edge)) {
        v = DiGraph_add_one_node(self, v_of_edge);
    } else {
        v = self.node_to_id[v_of_edge].cast<node_t>();
    }
    py::list items = py::list(edge_attr.attr("items")());
    self.adj[u][v] = node_attr_dict_factory();
    self.pred[v][u] = edge_attr_dict_factory();
    for (int i = 0; i < len(items); i++) {
        py::tuple kv = items[i].cast<py::tuple>();
        py::object pkey = kv[0];
        std::string weight_key = weight_to_string(pkey);
        weight_t value = kv[1].cast<weight_t>();
        self.adj[u][v].insert(std::make_pair(weight_key, value));
        self.pred[v][u].insert(std::make_pair(weight_key, value));
    }
}

py::object DiGraph_add_edge(py::args args, py::kwargs kwargs) {
    DiGraph& self = args[0].cast<DiGraph&>();
    self.dirty_nodes = true;
    self.dirty_adj = true;
    py::object u_of_edge = args[1], v_of_edge = args[2];
    py::dict edge_attr = kwargs;
    DiGraph_add_one_edge(self, u_of_edge, v_of_edge, edge_attr);
    return py::none();
}

py::object DiGraph_add_edges(DiGraph& self, py::list edges_for_adding, py::list edges_attr) {
    self.dirty_nodes = true;
    self.dirty_adj = true;
    if (py::len(edges_attr) != 0) {
        if (py::len(edges_for_adding) != py::len(edges_attr)) {
            PyErr_Format(PyExc_AssertionError, "Edges and Attributes lists must have same length.");
            return py::none();
        }
    }
    for (int i = 0; i < py::len(edges_for_adding); i++) {
        py::tuple one_edge_for_adding = edges_for_adding[i].cast<py::tuple>();
        py::dict edge_attr;
        if (py::len(edges_attr)) {
            edge_attr = edges_attr[i].cast<py::dict>();
        } else {
            edge_attr = py::dict();
        }
        DiGraph_add_one_edge(self, one_edge_for_adding[0], one_edge_for_adding[1], edge_attr);
    }
    return py::none();
}

py::object DiGraph_add_edges_from(py::args args, py::kwargs attr) {
    DiGraph& self = args[0].cast<DiGraph&>();
    self.dirty_nodes = true;
    self.dirty_adj = true;
    py::list ebunch_to_add = py::list(args[1]);
    for (int i = 0; i < len(ebunch_to_add); i++) {
        py::list e = py::list(ebunch_to_add[i]);
        py::object u, v;
        py::dict dd;
        switch (len(e)) {
            case 2: {
                u = e[0];
                v = e[1];
                break;
            }
            case 3: {
                u = e[0];
                v = e[1];
                dd = e[2].cast<py::dict>();
                break;
            }
            default: {
                PyErr_Format(PyExc_ValueError, "Edge tuple %R must be a 2 - tuple or 3 - tuple.", e.ptr());
                return py::none();
            }
        }
        node_t u_id, v_id;
        if (!self.node_to_id.contains(u)) {
            if (u.is_none()) {
                PyErr_Format(PyExc_ValueError, "None cannot be a node");
                return py::none();
            }
            u_id = DiGraph_add_one_node(self, u);

        } else {
            u_id = self.node_to_id[u].cast<node_t>();
        }
        if (!self.node_to_id.contains(v)) {
            if (v.is_none()) {
                PyErr_Format(PyExc_ValueError, "None cannot be a node");
                return py::none();
            }
            v_id = DiGraph_add_one_node(self, v);

        } else {
            v_id = self.node_to_id[v].cast<node_t>();
        }
        auto datadict = self.adj[u_id].count(v_id) ? self.adj[u_id][v_id] : node_attr_dict_factory();
        py::list items = py::list(attr.attr("items")());
        items.attr("extend")(py::list(dd.attr("items")()));
        for (int j = 0; j < py::len(items); j++) {
            py::tuple kv = items[j].cast<py::tuple>();
            py::object pkey = kv[0];
            std::string weight_key = weight_to_string(pkey);
            weight_t value = kv[1].cast<weight_t>();
            datadict.insert(std::make_pair(weight_key, value));
        }
        // Warning: in Graph.py the edge attr is directed assigned by the dict extended from the original attr
        self.adj[u_id][v_id].insert(datadict.begin(), datadict.end());
    }
    return py::none();
}

py::object DiGraph_add_edges_from_file(DiGraph& self, py::str file, py::object weighted, py::object is_transform) {
    self.dirty_nodes = true;
    self.dirty_adj = true;
    bool _is_transform = is_transform.cast<bool>();
    struct commactype : std::ctype<char> {
        commactype() : std::ctype<char>(get_table()) {}
        std::ctype_base::mask const* get_table() {
            std::ctype_base::mask* rc = 0;
            if (rc == 0) {
                rc = new std::ctype_base::mask[std::ctype<char>::table_size];
                std::fill_n(rc, std::ctype<char>::table_size, std::ctype_base::mask());
                rc[','] = std::ctype_base::space;
                rc[' '] = std::ctype_base::space;
                rc['\t'] = std::ctype_base::space;
                rc['\n'] = std::ctype_base::space;
                rc['\r'] = std::ctype_base::space;
            }
            return rc;
        }
    };

    std::ios::sync_with_stdio(0);
    std::string file_path = file.cast<std::string>();
    std::ifstream in;
    in.open(file_path);
    if (!in.is_open()) {
        PyErr_Format(PyExc_FileNotFoundError, "Please check the file and make sure the path only contains English");
        return py::none();
    }
    in.imbue(std::locale(std::locale(), new commactype));
    std::string data, key("weight");
    std::string su, sv;
    weight_t weight;
    while (in >> su >> sv) {
        py::str pu(su), pv(sv);
        node_t u, v;
        if (!self.node_to_id.contains(pu)) {
            u = DiGraph_add_one_node(self, pu);
        } else {
            u = self.node_to_id[pu].cast<node_t>();
        }
        if (!self.node_to_id.contains(pv)) {
            v = DiGraph_add_one_node(self, pv);
        } else {
            v = self.node_to_id[pv].cast<node_t>();
        }
        if (weighted.cast<bool>()) {
            in >> weight;
            self.adj[u][v][key] = weight;
            self.pred[v][u][key] = weight;
        } else {
            if (!self.adj[u].count(v)) {
                self.adj[u][v] = node_attr_dict_factory();
                self.pred[v][u] = node_attr_dict_factory();
            }
        }
    }
    if(_is_transform){
        Graph_L g_l = graph_to_linkgraph(self,true, key, true, false);
        self.linkgraph_structure = g_l;
        self.linkgraph_dirty = false;
    }
    in.close();
    return py::none();
}

py::object DiGraph_add_weighted_edge(DiGraph& self, py::object u_of_edge, py::object v_of_edge, weight_t weight) {
    self.dirty_nodes = true;
    self.dirty_adj = true;
    py::dict edge_attr;
    edge_attr["weight"] = weight;
    DiGraph_add_one_edge(self, u_of_edge, v_of_edge, edge_attr);
    return py::none();
}

py::object DiGraph_remove_edge(DiGraph& self, py::object u, py::object v) {
    self.dirty_nodes = true;
    self.dirty_adj = true;
    if (self.node_to_id.contains(u) && self.node_to_id.contains(v)) {
        node_t u_id = self.node_to_id[u].cast<node_t>();
        node_t v_id = self.node_to_id[v].cast<node_t>();
        auto& u_neighbors_info = self.adj[u_id];
        if (u_neighbors_info.find(v_id) != u_neighbors_info.end()) {
            u_neighbors_info.erase(v_id);
            auto& v_predecessors_info = self.pred[v_id];
            v_predecessors_info.erase(u_id);
            return py::none();
        }
    }
    PyErr_Format(PyExc_KeyError, "No edge %R-%R in graph.", u.ptr(), v.ptr());
    return py::none();
}

py::object DiGraph_remove_edges(py::object self, py::list edges_to_remove) {
    DiGraph& self_ = self.cast<DiGraph&>();
    for (int i = 0; i < py::len(edges_to_remove); i++) {
        py::tuple edge = edges_to_remove[i].cast<py::tuple>();
        py::object u = edge[0], v = edge[1];
        self.attr("remove_edge")(u, v);
    }
    self_.dirty_nodes = true;
    self_.dirty_adj = true;
    return py::none();
}

py::object DiGraph_remove_edges_from(py::object self, py::list ebunch) {
    DiGraph& self_ = self.cast<DiGraph&>();
    for (int i = 0; i < py::len(ebunch); i++) {
        py::tuple edge = ebunch[i].cast<py::tuple>();
        node_t u_id = edge[0].cast<node_t>();
        node_t v_id = edge[1].cast<node_t>();
        if (self_.adj[u_id].find(v_id) != self_.adj[u_id].end() && self_.adj[v_id].find(u_id) != self_.adj[v_id].end()) {
            self_.adj[u_id].erase(v_id);
            self_.pred[v_id].erase(u_id);
        }
    }
    return py::none();
}

py::object DiGraph_nodes_subgraph(py::object self, py::list from_nodes) {
    py::object G = self.attr("__class__")();
    Graph& self_ = self.cast<Graph&>();
    DiGraph& G_ = G.cast<DiGraph&>();
    G_.graph.attr("update")(self_.graph);
    py::object nodes = self.attr("nodes");
    py::object adj = self.attr("adj");
    for (int i = 0; i < py::len(from_nodes); i++) {
        py::object node = from_nodes[i];
        if (self_.node_to_id.contains(node)) {
            py::object node_attr = nodes[node];
            DiGraph_add_one_node(G_, node, node_attr);
        }
        py::object out_edges = adj[node];
        py::list edge_items = py::list(out_edges.attr("items")());
        for (int j = 0; j < py::len(edge_items); j++) {
            py::tuple item = edge_items[j].cast<py::tuple>();
            py::object v = item[0];
            py::object edge_attr = item[1];
            if (from_nodes.contains(v)) {
                DiGraph_add_one_edge(G_, node, v, edge_attr);
            }
        }
    }
    return G;
}

py::object DiGraph_generate_linkgraph(py::object self, py::object weight){
    DiGraph& G_ = self.cast<DiGraph&>();
    std::string w = weight_to_string(weight);
    Graph_L g_l = graph_to_linkgraph(G_, true, w, true, false);
    G_.linkgraph_dirty = false;
    G_.linkgraph_structure = g_l;
    return py::none();
}

py::object DiGraph_copy(py::object self) {
    DiGraph& self_ = self.cast<DiGraph&>();
    py::object G = self.attr("__class__")();
    DiGraph& G_ = G.cast<DiGraph&>();
    G_.graph.attr("update")(self_.graph);
    G_.id_to_node.attr("update")(self_.id_to_node);
    G_.node_to_id.attr("update")(self_.node_to_id);
    G_.node = self_.node;
    G_.adj = self_.adj;
    G_.pred = self_.pred;
    return py::object(G);
}

py::object DiGraph_is_directed(py::object self) {
    return py::cast(true);
}

py::object DiGraph_py(py::object self) {
    py::object G = py::module_::import("easygraph").attr("DiGraph")();
    G.attr("graph").attr("update")(self.attr("graph"));
    G.attr("adj").attr("update")(self.attr("adj"));
    G.attr("nodes").attr("update")(self.attr("nodes"));
    G.attr("pred").attr("update")(self.attr("pred"));
//    G.attr("succ").attr("update")(self.attr("succ"));
    return G;
}

py::object DiGraph::get_pred() {
    adj_dict_factory pred = this->pred;
    py::dict predecessors = py::dict();
    for (const auto& ego_edges : this->pred) {
        node_t start_point = ego_edges.first;
        py::dict ego_edges_dict = py::dict();
        for (const auto& edge_info : ego_edges.second) {
            node_t end_point = edge_info.first;
            const auto& edge_attr = edge_info.second;
            ego_edges_dict[this->id_to_node[py::cast(end_point)]] = attr_to_dict(edge_attr);
        }
        predecessors[this->id_to_node[py::cast(start_point)]] = ego_edges_dict;
    }

    return predecessors;
}

py::object DiGraph::get_edges() {
    py::list edges = py::list();
    std::set<std::pair<node_t, node_t> > seen;
    for (const auto& ego_edges : this->adj) {
        node_t u = ego_edges.first;
        for (const auto& edge_info : ego_edges.second) {
            node_t v = edge_info.first;
            const auto& edge_attr = edge_info.second;
            if (seen.find(std::make_pair(u, v)) == seen.end()) {
                seen.insert(std::make_pair(u, v));
                edges.append(py::make_tuple(this->id_to_node[py::cast(u)], this->id_to_node[py::cast(v)],
                                            attr_to_dict(edge_attr)));
            }
        }
    }
    return edges;
}

