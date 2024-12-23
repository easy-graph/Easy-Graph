#include "graph.h"
#include "linkgraph.h"
#include "../common/utils.h"


Graph::Graph() {
    this->id = 0;
    this->dirty_nodes = true;
    this->dirty_adj = true;
    this->linkgraph_dirty = true;
    this->csr_graph = nullptr;
    this->node_to_id = py::dict();
    this->id_to_node = py::dict();
    this->graph = py::dict();
    this->nodes_cache = py::dict();
    this->adj_cache = py::dict();
    this->coo_graph = nullptr;
}

py::object Graph__init__(py::args args, py::kwargs kwargs) {
    py::object self = args[0];
    self.attr("__init__")();
    Graph& self_ = self.cast<Graph&>();
    py::dict graph_attr = kwargs;
    self_.graph.attr("update")(graph_attr);
    self_.nodes_cache = py::dict();
    self_.adj_cache = py::dict();
    return py::none();
}

py::object Graph__iter__(py::object self) {
    return self.attr("nodes").attr("__iter__")();
}

py::object Graph__len__(py::object self) {
    Graph& self_ = self.cast<Graph&>();
    return py::cast(py::len(self_.node_to_id));
}

py::object Graph__contains__(py::object self, py::object node) {
    Graph& self_ = self.cast<Graph&>();
    try {
        return py::cast(self_.node_to_id.contains(node));
    } catch (const py::error_already_set&) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        if (PyErr_GivenExceptionMatches(PyExc_TypeError, type)) {
            return py::cast(false);
        } else {
            PyErr_Restore(type, value, traceback);
            return py::none();
        }
    }
}

py::object Graph__getitem__(py::object self, py::object node) {
    return self.attr("adj")[node];
}

node_t _add_one_node(Graph& self, py::object one_node_for_adding, py::object node_attr = py::dict()) {
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
    for (int i = 0; i < len(items); i++) {
        py::tuple kv = items[i].cast<py::tuple>();
        py::object pkey = kv[0];
        std::string weight_key = weight_to_string(pkey);
        weight_t value = kv[1].cast<weight_t>();
        self.node[id].insert(std::make_pair(weight_key, value));
    }
    return id;
}

py::object Graph_add_node(py::args args, py::kwargs kwargs) {
    Graph& self = args[0].cast<Graph&>();
    self.drop_cache();
    py::object one_node_for_adding = args[1];
    py::dict node_attr = kwargs;
    _add_one_node(self, one_node_for_adding, node_attr);
    return py::none();
}

py::object Graph_add_nodes(Graph& self, py::list nodes_for_adding, py::list nodes_attr) {
    self.drop_cache();
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
        _add_one_node(self, one_node_for_adding, node_attr);
    }
    return py::none();
}

py::object Graph_add_nodes_from(py::args args, py::kwargs kwargs) {
    Graph& self = args[0].cast<Graph&>();
    self.drop_cache();
    py::list nodes_for_adding = py::list(args[1]);
    for (int i = 0; i < py::len(nodes_for_adding); i++) {
        bool newnode;
        py::dict attr = kwargs;
        py::dict newdict, ndict;
        py::object n = nodes_for_adding[i];
        try {
            newnode = !self.node_to_id.contains(n);
            newdict = attr;
        }
        catch (const py::error_already_set&) {
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
            _add_one_node(self, n);
        }
        node_t id = self.node_to_id[n].cast<node_t>();
        for (auto item : newdict) {
            std::string weight_key = weight_to_string(item.first.cast<py::object>());
            weight_t value = item.second.cast<weight_t>();
            self.node[id].insert(std::make_pair(weight_key, value));
        }
    }
    return py::none();
}

py::object Graph_remove_node(Graph& self, py::object node_to_remove) {
    self.drop_cache();
    if (!self.node_to_id.contains(node_to_remove)) {
        PyErr_Format(PyExc_KeyError, "No node %R in graph.", node_to_remove.ptr());
        return py::none();
    }
    node_t node_id = self.node_to_id[node_to_remove].cast<node_t>();
    for (const auto& neighbor_info : self.adj[node_id]) {
        node_t neighbor_id = neighbor_info.first;
        self.adj[neighbor_id].erase(node_id);
    }
    self.adj.erase(node_id);
    self.node.erase(node_id);
    self.node_to_id.attr("pop")(node_to_remove);
    self.id_to_node.attr("pop")(node_id);
    return py::none();
}

py::object Graph_remove_nodes(py::object self, py::list nodes_to_remove) {
    Graph& self_ = self.cast<Graph&>();
    self_.drop_cache();
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

py::object Graph_number_of_nodes(Graph& self) {
    return py::cast(int(self.node.size()));
}

py::object Graph_has_node(Graph& self, py::object node) {
    return py::cast(self.node_to_id.contains(node));
}

py::object Graph_nbunch_iter(py::object self, py::object nbunch) {
    py::object bunch = py::none();
    if (nbunch.is_none()) {
        bunch = self.attr("adj").attr("__iter__")();
    } else if (self.contains(nbunch)) {
        py::list nbunch_wrapper = py::list();
        nbunch_wrapper.append(nbunch);
        bunch = nbunch_wrapper.attr("__iter__")();
    } else {
        py::list nbunch_list = py::list(nbunch), nodes_list = py::list();
        for (int i = 0; i < py::len(nbunch_list); i++) {
            py::object n = nbunch_list[i];
            if (self.contains(n)) {
                nodes_list.append(n);
            }
        }
        bunch = nbunch_list.attr("__iter__")();
    }
    return bunch;
}

void _add_one_edge(Graph& self, py::object u_of_edge, py::object v_of_edge, py::object edge_attr) {
    node_t u, v;
    if (!self.node_to_id.contains(u_of_edge)) {
        u = _add_one_node(self, u_of_edge);
    } else {
        u = self.node_to_id[u_of_edge].cast<node_t>();
    }
    if (!self.node_to_id.contains(v_of_edge)) {
        v = _add_one_node(self, v_of_edge);
    } else {
        v = self.node_to_id[v_of_edge].cast<node_t>();
    }
    py::list items = py::list(edge_attr.attr("items")());
    self.adj[u][v] = node_attr_dict_factory();
    self.adj[v][u] = node_attr_dict_factory();
    for (int i = 0; i < len(items); i++) {
        py::tuple kv = items[i].cast<py::tuple>();
        py::object pkey = kv[0];
        std::string weight_key = weight_to_string(pkey);
        weight_t value = kv[1].cast<weight_t>();
        self.adj[u][v].insert(std::make_pair(weight_key, value));
        self.adj[v][u].insert(std::make_pair(weight_key, value));
    }
}

py::object Graph_add_edge(py::args args, py::kwargs kwargs) {
    Graph& self = args[0].cast<Graph&>();
    self.drop_cache();
    py::object u_of_edge = args[1], v_of_edge = args[2];
    py::dict edge_attr = kwargs;
    _add_one_edge(self, u_of_edge, v_of_edge, edge_attr);
    return py::none();
}

py::object Graph_add_edges(Graph& self, py::list edges_for_adding, py::list edges_attr) {
    self.drop_cache();
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
        _add_one_edge(self, one_edge_for_adding[0], one_edge_for_adding[1], edge_attr);
    }
    return py::none();
}

py::object Graph_add_edges_from(py::args args, py::kwargs attr) {
    Graph& self = args[0].cast<Graph&>();
    self.drop_cache();
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
            u_id = _add_one_node(self, u);
        } else {
            u_id = self.node_to_id[u].cast<node_t>();
        }
        if (!self.node_to_id.contains(v)) {
            if (v.is_none()) {
                PyErr_Format(PyExc_ValueError, "None cannot be a node");
                return py::none();
            }
            v_id = _add_one_node(self, v);
        } else {
            v_id = (self.node_to_id[v]).cast<node_t>();
        }
        auto datadict = self.adj[u_id].count(v_id) ? self.adj[u_id][v_id] : node_attr_dict_factory();
        py::list items = py::list(attr.attr("items")());
        items.attr("extend")(py::list(dd.attr("items")()));
        for (int i = 0; i < py::len(items); i++) {
            py::tuple kv = items[i].cast<py::tuple>();
            py::object pkey = kv[0];
            std::string weight_key = weight_to_string(pkey);
            weight_t value = kv[1].cast<weight_t>();
            datadict.insert(std::make_pair(weight_key, value));
        }
        // Warning: in Graph.py the edge attr is directed assigned by the dict extended from the original attr
        self.adj[u_id][v_id].insert(datadict.begin(), datadict.end());
        self.adj[v_id][u_id].insert(datadict.begin(), datadict.end());
    }
    return py::none();
}

py::object Graph_add_edges_from_file(Graph& self, py::str file, py::object weighted, py::object is_transform) {
    self.drop_cache();
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
                rc['	'] = std::ctype_base::space;
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
            u = _add_one_node(self, pu);
        } else {
            u = self.node_to_id[pu].cast<node_t>();
        }
        if (!self.node_to_id.contains(pv)) {
            v = _add_one_node(self, pv);
        } else {
            v = self.node_to_id[pv].cast<node_t>();
        }
        if (weighted.cast<bool>()) {
            in >> weight;
            self.adj[u][v][key] = self.adj[v][u][key] = weight;
        } else {
            if (!self.adj[u].count(v)) {
                self.adj[u][v] = node_attr_dict_factory();
            }
            if (!self.adj[v].count(u)) {
                self.adj[v][u] = node_attr_dict_factory();
            }
        }
    }
    in.close();
    if(_is_transform){
        Graph_L g_l = graph_to_linkgraph(self, false, key, true, false);
        self.linkgraph_structure = g_l;
        self.linkgraph_dirty = false;
    }
    return py::none();
}

py::object Graph_add_weighted_edge(Graph& self, py::object u_of_edge, py::object v_of_edge, weight_t weight) {
    self.drop_cache();
    py::dict edge_attr;
    edge_attr["weight"] = weight;
    _add_one_edge(self, u_of_edge, v_of_edge, edge_attr);
    return py::none();
}

py::object Graph_remove_edge(Graph& self, py::object u, py::object v) {
    self.drop_cache();
    if (self.node_to_id.contains(u) && self.node_to_id.contains(v)) {
        node_t u_id = self.node_to_id[u].cast<node_t>();
        node_t v_id = self.node_to_id[v].cast<node_t>();
        auto& v_neighbors_info = self.adj[u_id];
        if (v_neighbors_info.find(v_id) != v_neighbors_info.end()) {
            v_neighbors_info.erase(v_id);
            if (u_id != v_id) {
                self.adj[v_id].erase(u_id);
            }
            return py::none();
        }
    }
    PyErr_Format(PyExc_KeyError, "No edge %R-%R in graph.", u.ptr(), v.ptr());
    return py::none();
}

py::object Graph_remove_edges(py::object self, py::list edges_to_remove) {
    Graph& self_ = self.cast<Graph&>();
    for (int i = 0; i < py::len(edges_to_remove); i++) {
        py::tuple edge = edges_to_remove[i].cast<py::tuple>();
        py::object u = edge[0], v = edge[1];
        self.attr("remove_edge")(u, v);
    }
    self_.drop_cache();
    return py::none();
}

py::object Graph_number_of_edges(py::object self, py::object u, py::object v) {
    if (u.is_none()) {
        return self.attr("size")();
    }
    Graph& self_ = self.cast<Graph&>();
    node_t u_id = self_.node_to_id.attr("get")(u, -1).cast<node_t>();
    node_t v_id = self_.node_to_id.attr("get")(v, -1).cast<node_t>();
    return py::cast(int(self_.adj.count(u_id) && self_.adj[u_id].count(v_id)));
}

py::object Graph_has_edge(Graph& self, py::object u, py::object v) {
    if (self.node_to_id.contains(u) && self.node_to_id.contains(v)) {
        node_t u_id = self.node_to_id[u].cast<node_t>();
        node_t v_id = self.node_to_id[v].cast<node_t>();
        auto& v_neighbors_info = self.adj[u_id];
        if (v_neighbors_info.find(v_id) != v_neighbors_info.end()) {
            return py::cast(true);
        }
    }
    return py::cast(false);
}

py::object Graph_copy(py::object self) {
    Graph& self_ = self.cast<Graph&>();
    py::object G = self.attr("__class__")();
    Graph& G_ = G.cast<Graph&>();
    G_.graph.attr("update")(self_.graph);
    G_.id_to_node.attr("update")(self_.id_to_node);
    G_.node_to_id.attr("update")(self_.node_to_id);
    G_.id = self_.id;
    G_.node = self_.node;
    G_.adj = self_.adj;
    return G;
}

py::object Graph_degree(py::object self, py::object weight) {
    py::dict degree;
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

py::object Graph_neighbors(py::object self, py::object node) {
    Graph& self_ = self.cast<Graph&>();
    if (self_.node_to_id.contains(node)) {
        return self.attr("adj")[node].attr("__iter__")();
    } else {
        PyErr_Format(PyExc_KeyError, "No node %R", node.ptr());
        return py::none();
    }
}

py::object Graph_generate_linkgraph(py::object self, py::object weight){
    Graph& G_ = self.cast<Graph&>();
    std::string w = weight_to_string(weight);
    Graph_L g_l = graph_to_linkgraph(G_, false, w, true, false);
    G_.linkgraph_dirty = false;
    G_.linkgraph_structure = g_l;
    return py::none();
}

py::object Graph_nodes_subgraph(py::object self, py::list from_nodes) {
    py::object G = self.attr("__class__")();
    Graph& self_ = self.cast<Graph&>();
    Graph& G_ = G.cast<Graph&>();
    G_.graph.attr("update")(self_.graph);
    py::object nodes = self.attr("nodes");
    py::object adj = self.attr("adj");
    for (int i = 0; i < py::len(from_nodes); i++) {
        py::object node = from_nodes[i];
        if (self_.node_to_id.contains(node)) {
            py::object node_attr = nodes[node];
            _add_one_node(G_, node, node_attr);
        }
        py::object out_edges = adj[node];
        py::list edge_items = py::list(out_edges.attr("items")());
        for (int j = 0; j < py::len(edge_items); j++) {
            py::tuple item = edge_items[j].cast<py::tuple>();
            py::object v = item[0];
            py::object edge_attr = item[1];
            if (from_nodes.contains(v)) {
                _add_one_edge(G_, node, v, edge_attr);
            }
        }
    }
    return G;
}

py::object Graph_ego_subgraph(py::object self, py::object center) {
    py::list neighbors_of_center = py::list(self.attr("all_neighbors")(center));
    neighbors_of_center.append(center);
    return self.attr("nodes_subgraph")(neighbors_of_center);
}

py::object Graph_size(py::object self, py::object weight) {
    py::dict degree = self.attr("degree")(weight).cast<py::dict>();
    weight_t s = 0;
    for (auto item : degree) {
        s += item.second.cast<weight_t>();
    }
    return (weight.is_none()) ? py::cast(int(s) / 2) : py::cast(s / 2);
}

py::object Graph_is_directed(py::object self) {
    return py::cast(false);
}

py::object Graph_is_multigraph(py::object self) {
    return py::cast(false);
}

py::object Graph_to_index_node_graph(py::object self, py::object begin_index) {
    py::object G = self.attr("__class__")();
    G.attr("graph").attr("update")(self.attr("graph"));
    py::dict index_of_node = py::dict(), node_of_index = py::dict();
    int begin = begin_index.cast<int>();
    int index = 0;
    for (auto item : self.attr("nodes").cast<py::dict>()) {
        py::object node = item.first.cast<py::object>();
        py::dict node_attr = item.second.cast<py::dict>();
        G.attr("add_node")(py::cast(index + begin), **node_attr);
        index_of_node[node] = index + begin;
        node_of_index[py::cast(index + begin)] = node;
        index++;
    }
    for (auto item : self.attr("adj").cast<py::dict>()) {
        py::object u = item.first.cast<py::object>();
        py::dict nbrs = item.second.cast<py::dict>();
        for (auto item_ : nbrs) {
            py::object v = item_.first.cast<py::object>();
            py::dict edge_data = item_.second.cast<py::dict>();
            G.attr("add_edge")(index_of_node[u], index_of_node[v], **edge_data);
        }
    }
    return py::make_tuple(G, index_of_node, node_of_index);
}

py::object Graph_py(py::object self) {
    py::object G = py::module_::import("easygraph").attr("Graph")();
    G.attr("graph").attr("update")(self.attr("graph"));
    G.attr("adj").attr("update")(self.attr("adj"));
    G.attr("nodes").attr("update")(self.attr("nodes"));
    return G;
}

py::object Graph::get_nodes() {
    if (this->dirty_nodes) {
        py::dict nodes = py::dict();
        for (const auto& node_info : node) {
            node_t id = node_info.first;
            const auto& node_attr = node_info.second;
            nodes[this->id_to_node[py::cast(id)]] = attr_to_dict(node_attr);
        }
        this->nodes_cache = nodes;
        this->dirty_nodes = false;
    }
    return this->nodes_cache;
}

py::object Graph::get_name() {
    return this->graph.attr("get")("name", "");
}

py::object Graph::set_name(py::object name) {
    this->graph[py::cast("name")] = name;
    return py::none();
}

py::object Graph::get_node_index() {
    py::dict node_index = py::dict();
    int len = py::len(this->node_to_id);
    for(int i = 1; i <= len; i++){
        node_index[this->id_to_node[py::cast(i)]] = py::cast(i - 1);
    }
    return node_index;
}
py::object Graph::get_graph() {
    return this->graph;
}

py::object Graph::get_adj() {
    if (this->dirty_adj) {
        py::dict adj = py::dict();
        for (const auto& ego_edges : this->adj) {
            node_t start_point = ego_edges.first;
            py::dict ego_edges_dict = py::dict();
            for (const auto& edge_info : ego_edges.second) {
                node_t end_point = edge_info.first;
                const auto& edge_attr = edge_info.second;
                ego_edges_dict[this->id_to_node[py::cast(end_point)]] = attr_to_dict(edge_attr);
            }
            adj[this->id_to_node[py::cast(start_point)]] = ego_edges_dict;
        }
        this->adj_cache = adj;
        this->dirty_adj = false;
    }
    return this->adj_cache;
}

py::object Graph::get_edges() {
    py::list edges = py::list();
    std::set<std::pair<node_t, node_t> > seen;
    for (const auto& ego_edges : this->adj) {
        node_t u = ego_edges.first;
        for (const auto& edge_info : ego_edges.second) {
            node_t v = edge_info.first;
            const auto& edge_attr = edge_info.second;
            if (seen.find(std::make_pair(u, v)) == seen.end()) {
                seen.insert(std::make_pair(u, v));
                seen.insert(std::make_pair(v, u));
                edges.append(py::make_tuple(this->id_to_node[py::cast(u)], this->id_to_node[py::cast(v)], attr_to_dict(edge_attr)));
            }
        }
    }
    return edges;
}


Graph_L Graph::_get_linkgraph_structure()  {
    return this->linkgraph_structure;
}
bool Graph::is_linkgraph_dirty(){
    return this->linkgraph_dirty;
}
std::vector<graph_edge> Graph::_get_edges(bool if_directed) {
    std::vector<graph_edge> edges;
    std::set<std::pair<node_t, node_t> > seen;
    for (const auto& ego_edges : this->adj) {
        node_t u = ego_edges.first;
        for (const auto& edge_info : ego_edges.second) {
            node_t v = edge_info.first;
            const auto& edge_attr = edge_info.second;
            if (seen.find(std::make_pair(u, v)) == seen.end()) {
                
                seen.insert(std::make_pair(u, v));
                if(!if_directed){
                    seen.insert(std::make_pair(v, u));
                }
                
                edges.emplace_back(u, v, edge_attr);
            }
        }
    }
    return edges;
}

void Graph::drop_cache() {
    dirty_nodes = true;
    dirty_adj = true;
    linkgraph_dirty = true;
    csr_graph = nullptr;
}

std::shared_ptr<CSRGraph> Graph::gen_CSR(const std::string& weight) {
    if (csr_graph != nullptr) {
        if (csr_graph->W_map.find(weight) == csr_graph->W_map.end()) {
            auto W = std::make_shared<std::vector<double>>();

            // According to C++ Standard, the iteration order of a unordered contrainer will
            // not change without rehashing which only happens during a insert.
            for (node_t n : csr_graph->nodes) {
                // if n is not in adj, this way can raise an exception
                const auto& n_adjs = adj.find(n)->second;

                for (auto adj_it = n_adjs.begin(); adj_it != n_adjs.end(); ++adj_it) {
                    const edge_attr_dict_factory& edge_attr = adj_it->second;
                    auto edge_it = edge_attr.find(weight);
                    weight_t w = edge_it != edge_attr.end() ? edge_it->second : 1.0;

                    W->push_back(w);
                }
            }

            csr_graph->W_map[weight] = W;
        }
    } else {
        // the graph has been modified

        csr_graph = std::make_shared<CSRGraph>();

        std::vector<node_t>& nodes = csr_graph->nodes;
        for (auto it = node.begin(); it != node.end(); ++it) {
            nodes.push_back(it->first);
        }

        std::sort(nodes.begin(), nodes.end());

        std::unordered_map<node_t, int>& node2idx = csr_graph->node2idx;

        for (int i = 0; i < nodes.size(); ++i) {
            node2idx[nodes[i]] = i;
        }

        std::vector<int>& V = csr_graph->V;
        std::vector<int>& E = csr_graph->E;
        auto W = std::make_shared<std::vector<double>>();

        for (int idx = 0; idx < nodes.size(); ++idx) {
            V.push_back(E.size());

            node_t n = nodes[idx];

            // if n is not in adj, this way can raise an exception
            const auto& n_adjs = adj.find(n)->second;

            for (auto adj_it = n_adjs.begin(); adj_it != n_adjs.end(); ++adj_it) {
                const edge_attr_dict_factory& edge_attr = adj_it->second;
                auto edge_it = edge_attr.find(weight);
                weight_t w = edge_it != edge_attr.end() ? edge_it->second : 1.0;

                W->push_back(w);
                E.push_back(node2idx[adj_it->first]);
            }
        }

        V.push_back(E.size());

        csr_graph->W_map[weight] = W;
    }

    return csr_graph;
}

std::shared_ptr<CSRGraph> Graph::gen_CSR() {
    if (csr_graph != nullptr) {
        if (csr_graph->unweighted_W.size() != csr_graph->E.size()) {
            csr_graph->unweighted_W = std::vector<double>(csr_graph->E.size(), 1.0);
        }
    } else {
        // the graph has been modified

        csr_graph = std::make_shared<CSRGraph>();

        std::vector<node_t>& nodes = csr_graph->nodes;
        for (auto it = node.begin(); it != node.end(); ++it) {
            nodes.push_back(it->first);
        }

        std::sort(nodes.begin(), nodes.end());

        std::unordered_map<node_t, int>& node2idx = csr_graph->node2idx;

        for (int i = 0; i < nodes.size(); ++i) {
            node2idx[nodes[i]] = i;
        }

        std::vector<int>& V = csr_graph->V;
        std::vector<int>& E = csr_graph->E;

        for (int idx = 0; idx < nodes.size(); ++idx) {
            V.push_back(E.size());

            node_t n = nodes[idx];

            // if n is not in adj, this way can raise an exception
            const auto& n_adjs = adj.find(n)->second;

            for (auto adj_it = n_adjs.begin(); adj_it != n_adjs.end(); ++adj_it) {
                const edge_attr_dict_factory& edge_attr = adj_it->second;

                E.push_back(node2idx[adj_it->first]);
            }
        }

        V.push_back(E.size());
        csr_graph->unweighted_W = std::vector<double>(E.size(), 1.0);
    }

    return csr_graph;
}

std::shared_ptr<std::vector<int>> Graph::gen_CSR_sources(const py::object& py_sources) {
    auto sources = std::make_shared<std::vector<int>>();

    if (py_sources.is_none()) {
        for (int i = 0; i < csr_graph->V.size() - 1; ++i) {
            sources->push_back(i);
        }
    } else {
        for (auto it = py_sources.begin(); it != py_sources.end(); ++it) {
            sources->push_back(csr_graph->node2idx[node_to_id[*it].cast<node_t>()]);
        }
    }

    return sources;
}

std::shared_ptr<COOGraph> Graph::gen_COO() {
    if (coo_graph != nullptr) {
        if (coo_graph->unweighted_W.size() != coo_graph->row.size()) {
            coo_graph->unweighted_W = std::vector<double>(coo_graph->row.size(), 1.0);
        }
    } else {

        coo_graph = std::make_shared<COOGraph>();

        std::vector<node_t>& nodes = coo_graph->nodes;
        for (auto it = node.begin(); it != node.end(); ++it) {
            nodes.push_back(it->first);
        }

        std::sort(nodes.begin(), nodes.end());

        std::unordered_map<node_t, int>& node2idx = coo_graph->node2idx;
        for (int i = 0; i < nodes.size(); ++i) {
            node2idx[nodes[i]] = i;
        }

        std::vector<int>& row = coo_graph->row;
        std::vector<int>& col = coo_graph->col;
        std::vector<double>& unweighted_W = coo_graph->unweighted_W;

        for (int idx = 0; idx < nodes.size(); ++idx) {
            node_t n = nodes[idx];

            const auto& n_adjs = adj.find(n)->second;

            for (auto adj_it = n_adjs.begin(); adj_it != n_adjs.end(); ++adj_it) {
                node_t neighbor = adj_it->first;

                row.push_back(idx);            
                col.push_back(node2idx[neighbor]);

                unweighted_W.push_back(1.0);
            }
        }
    }

    return coo_graph;
}

std::shared_ptr<COOGraph> Graph::gen_COO(const std::string& weight) {
    if (coo_graph != nullptr) {
        if (coo_graph->W_map.find(weight) == coo_graph->W_map.end()) {
            auto W = std::make_shared<std::vector<double>>();

            for (node_t n : coo_graph->nodes) {
                const auto& n_adjs = adj.find(n)->second;

                for (auto adj_it = n_adjs.begin(); adj_it != n_adjs.end(); ++adj_it) {
                    const edge_attr_dict_factory& edge_attr = adj_it->second;
                    auto edge_it = edge_attr.find(weight);
                    weight_t w = edge_it != edge_attr.end() ? edge_it->second : 1.0;

                    W->push_back(w);
                }
            }

            coo_graph->W_map[weight] = W;
        }
    } else {
        coo_graph = std::make_shared<COOGraph>();

        std::vector<node_t>& nodes = coo_graph->nodes;
        for (auto it = node.begin(); it != node.end(); ++it) {
            nodes.push_back(it->first);
        }

        std::sort(nodes.begin(), nodes.end());

        std::unordered_map<node_t, int>& node2idx = coo_graph->node2idx;
        for (int i = 0; i < nodes.size(); ++i) {
            node2idx[nodes[i]] = i;
        }

        std::vector<int>& row = coo_graph->row;
        std::vector<int>& col = coo_graph->col;
        auto W = std::make_shared<std::vector<double>>();

        for (int idx = 0; idx < nodes.size(); ++idx) {
            node_t n = nodes[idx];

            const auto& n_adjs = adj.find(n)->second;

            for (auto adj_it = n_adjs.begin(); adj_it != n_adjs.end(); ++adj_it) {
                const edge_attr_dict_factory& edge_attr = adj_it->second;
                auto edge_it = edge_attr.find(weight);
                weight_t w = edge_it != edge_attr.end() ? edge_it->second : 1.0;

                row.push_back(idx);
                col.push_back(node2idx[adj_it->first]); 

                W->push_back(w);
            }
        }

        coo_graph->W_map[weight] = W;
    }

    return coo_graph;
}

std::shared_ptr<COOGraph> Graph::transfer_csr_to_coo(const std::shared_ptr<CSRGraph>& csr_graph) {
    auto coo_graph = std::make_shared<COOGraph>();

    coo_graph->nodes = csr_graph->nodes;
    coo_graph->node2idx = csr_graph->node2idx;

    const std::vector<int>& V = csr_graph->V;
    const std::vector<int>& E = csr_graph->E;
    int num_edges = E.size();

    coo_graph->row.reserve(num_edges);
    coo_graph->col.reserve(num_edges);

    std::vector<int>& row = coo_graph->row;
    std::vector<int>& col = coo_graph->col;

    for (int i = 0; i < V.size() - 1; ++i) {
        int start_idx = V[i];
        int end_idx = V[i + 1];

        for (int j = start_idx; j < end_idx; ++j) {
            row.push_back(i);      
            col.push_back(E[j]);   
        }
    }

    if (!csr_graph->unweighted_W.empty()) {
        coo_graph->unweighted_W = csr_graph->unweighted_W;
    } else {
        coo_graph->W_map = csr_graph->W_map;
    }

    return coo_graph;
}