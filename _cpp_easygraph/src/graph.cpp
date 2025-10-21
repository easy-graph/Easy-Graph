#include "graph.h"
#include "common.h"
#include "utils.h"

Graph::Graph() {
	py::object MappingProxyType = py::module_::import("types").attr("MappingProxyType");
	this->id = 0;
	this->dirty_nodes = true;
	this->dirty_adj = true;
	this->node_to_id = py::dict();
	this->id_to_node = py::dict();
	this->graph = py::dict();
	this->nodes_cache = MappingProxyType(py::dict());
	this->adj_cache = MappingProxyType(py::dict());
}

void Graph___init__(Graph &self, py::kwargs kwargs) {
	// 可选：防止重复初始化，placement-new 复位
	new (&self) Graph();

	// graph.update(kwargs)
	if (kwargs && py::len(kwargs) > 0) {
		self.graph.attr("update")(kwargs);
		// 也可手动拷贝键值：
		// for (auto &item : kwargs) self.graph[py::str(item.first)] = item.second;
	}

	// 重新创建只读 cache
	py::object MappingProxyType = py::module_::import("types").attr("MappingProxyType");
	self.nodes_cache = MappingProxyType(py::dict());
	self.adj_cache   = MappingProxyType(py::dict());
}

py::object Graph__iter__(py::object self) {
	return self.attr("nodes").attr("__iter__")();
}

py::object Graph__len__(const py::object& self) {
	// 将 self 转成 Graph&
	Graph& self_ = self.cast<Graph&>();

	// 计算 Python 层 len(node_to_id)
	py::ssize_t n = py::len(self_.node_to_id);

	// 返回 Python int
	return py::int_(n);
}

py::object Graph__contains__(const py::object& self, const py::object& node) {
	Graph& self_ = self.cast<Graph&>();

	try {
		// 若 node_to_id 是 Python dict，可直接调用 contains()
		bool has = self_.node_to_id.attr("__contains__")(node).cast<bool>();
		return py::bool_(has);
	}
	catch (const py::error_already_set& e) {
		// 捕获 Python 异常
		if (e.matches(PyExc_TypeError)) {
			// TypeError: 节点类型不匹配 → 返回 False
			PyErr_Clear();
			return py::bool_(false);
		} else {
			// 重新抛出其他异常，让 Python 层能看到原始错误
			throw;
		}
	}
}

py::object Graph__getitem__(py::object self, py::object node) {
	return self.attr("adj")[node];
}

node_t _add_one_node(Graph& self,
							const py::object& one_node_for_adding,
							py::object node_attr /* = None */) {
	if (node_attr.is_none()) {
		node_attr = py::dict();
	}

	node_t id;
	// if self.node_to_id.contains(one_node_for_adding)
	const bool has =
		self.node_to_id.attr("__contains__")(one_node_for_adding).cast<bool>();

	if (has) {
		// id = self.node_to_id[one_node_for_adding]
		id = self.node_to_id.attr("__getitem__")(one_node_for_adding).cast<node_t>();
	} else {
		id = ++(self.id);

		// self.id_to_node[id] = one_node_for_adding
		self.id_to_node.attr("__setitem__")(py::int_(id), one_node_for_adding);

		// self.node_to_id[one_node_for_adding] = id
		self.node_to_id.attr("__setitem__")(one_node_for_adding, py::int_(id));
	}

	// items = list(node_attr.items())
	py::list items = py::list(node_attr.attr("items")());

	// 初始化 C++ 侧的属性/邻接容器
	self.node[id] = node_attr_dict_factory();
	self.adj[id]  = adj_attr_dict_factory();

	for (py::ssize_t i = 0; i < py::len(items); ++i) {
		py::tuple kv = items[i].cast<py::tuple>();
		py::object pkey = kv[0];
		std::string weight_key = weight_to_string(pkey);
		weight_t value = kv[1].cast<weight_t>();
		self.node[id].insert({weight_key, value});
	}

	return id;
}

py::object Graph_add_node(const py::tuple& args, const py::dict& kwargs) {
	// args: [self, node]
	Graph& self = args[0].cast<Graph&>();
	py::object one_node_for_adding = args[1];

	self.dirty_nodes = true;
	self.dirty_adj   = true;

	// kwargs 直接当作属性字典传入
	py::object node_attr = kwargs;   // kwargs 本身就是 dict-like
	_add_one_node(self, one_node_for_adding, node_attr);

	return py::none();               // 等价于原来的 py::object()
}


py::object Graph_add_nodes(Graph& self,
								  const py::list& nodes_for_adding,
								  const py::list& nodes_attr) {
	self.dirty_nodes = true;
	self.dirty_adj   = true;

	if (py::len(nodes_attr) != 0 &&
		py::len(nodes_for_adding) != py::len(nodes_attr)) {
		throw py::value_error("Nodes and attributes lists must have the same length.");
		}

	for (py::ssize_t i = 0; i < py::len(nodes_for_adding); ++i) {
		py::object one_node_for_adding = nodes_for_adding[i];

		py::dict node_attr;
		if (py::len(nodes_attr) != 0) {
			node_attr = nodes_attr[i].cast<py::dict>();  // 若不是 dict，会抛 TypeError
		} else {
			node_attr = py::dict();
		}

		_add_one_node(self, one_node_for_adding, node_attr);
	}

	return py::none();
}

void Graph_add_nodes_from(Graph& self, const py::iterable& nodes_for_adding, const py::kwargs& kwargs) {
    self.dirty_nodes = true;
    self.dirty_adj   = true;

    // 预先把 kwargs 复制到一个 dict（避免原地修改 kwargs）
    py::dict base_attrs;
    for (auto item : kwargs) {
        base_attrs[item.first] = item.second;
    }

    for (py::handle h : nodes_for_adding) {
        py::object n_obj = py::reinterpret_borrow<py::object>(h);
        py::dict    merged_attrs = py::dict(base_attrs);  // 每个节点的属性从 kwargs 起步

        // 支持形如 (node, {attr...}) 的二元组
        if (py::isinstance<py::tuple>(n_obj)) {
            py::tuple t = n_obj.cast<py::tuple>();
            if (py::len(t) != 2)
                throw py::type_error("Each tuple must be (node, dict).");

            py::object node = t[0];
            py::object maybe_dict = t[1];

            if (!py::isinstance<py::dict>(maybe_dict))
                throw py::type_error("Second element of tuple must be a dict.");

            py::dict ndict = maybe_dict.cast<py::dict>();
            // 合并 per-node 属性：kwargs < ndict（ndict 覆盖同名键）
            for (auto kv : ndict) {
                merged_attrs[kv.first] = kv.second;
            }
            n_obj = std::move(node);
        }

        // 禁止 None 作为节点
        if (n_obj.is_none())
            throw py::value_error("None cannot be a node.");

        // 如为新节点则先添加
        bool newnode = !self.node_to_id.contains(n_obj);
        if (newnode) {
             _add_one_node(self, n_obj, py::none());
        }

        // 获取 node id；若你的 map 直接存的是 node_t，可去掉 cast
        node_t id = py::cast<node_t>(self.node_to_id[n_obj]);

        // 写入属性
        for (auto kv : merged_attrs) {
            py::handle k = kv.first;
            py::handle v = kv.second;

            // 若你的 weight_to_string 接受 py::object：
            std::string weight_key = weight_to_string(py::reinterpret_borrow<py::object>(k));
            weight_t    weight_val = py::cast<weight_t>(v);

            self.node[id].insert(std::make_pair(std::move(weight_key), std::move(weight_val)));
        }
    }
}
void Graph_remove_node(Graph& self, const py::object& node_to_remove) {
	self.dirty_nodes = true;
	self.dirty_adj   = true;

	if (!self.node_to_id.contains(node_to_remove)) {
		// KeyError: 带上对象 repr
		throw py::key_error(py::str("No node {} in graph.").format(node_to_remove));
	}

	// 从 Python dict 中取出 id 并转为 node_t
	node_t node_id = py::cast<node_t>(self.node_to_id[node_to_remove]);

	// 先从所有邻居的邻接表中删除该节点
	// 注：这里修改的是 neighbor 的容器，不会使对 self.adj[node_id] 的遍历失效
	for (const auto& kv : self.adj[node_id]) {
		const node_t neighbor_id = kv.first;
		self.adj[neighbor_id].erase(node_id);
	}

	// 再删掉该节点的邻接表与节点属性
	self.adj.erase(node_id);
	self.node.erase(node_id);

	// 同步 Python 侧的映射（与原代码一致使用 pop）
	self.node_to_id.attr("pop")(node_to_remove);
	self.id_to_node.attr("pop")(node_id);
}

void Graph_remove_nodes(Graph& self, const py::sequence& nodes_to_remove) {
	self.dirty_nodes = true;
	self.dirty_adj   = true;

	const std::size_t n = py::len(nodes_to_remove);

	// 先整体校验（与原逻辑一致：任一不存在则立即报错并不做部分删除）
	for (std::size_t i = 0; i < n; ++i) {
		py::object node = nodes_to_remove[i];
		if (!self.node_to_id.contains(node)) {
			throw py::key_error(py::str("No node {} in graph.").format(node));
		}
	}

	// 再逐个删除（若输入包含重复节点，第二次删除会抛 KeyError，与原行为一致）
	for (std::size_t i = 0; i < n; ++i) {
		py::object node = nodes_to_remove[i];
		Graph_remove_node(self, node);  // 直接调 C++ 实现，避免 Python 回调开销
	}
}

py::object Graph_number_of_nodes(Graph& self) {
	return py::int_(self.node.size());
}

bool Graph_has_node(Graph &self, py::object node) {
	return self.node_to_id.contains(node);
}

py::object Graph_nbunch_iter(py::object self, py::object nbunch) {
	py::object bunch = py::object();
	if (nbunch.is_none()) {
		bunch = self.attr("adj").attr("__iter__")();
	}
	else if (self.contains(nbunch)) {
		py::list nbunch_wrapper = py::list();
		nbunch_wrapper.append(nbunch);
		bunch = nbunch_wrapper.attr("__iter__")();
	}
	else {
		py::list nbunch_list = py::list(nbunch), nodes_list = py::list();
		for (int i = 0;i < py::len(nbunch_list);i++) {
			py::object n = nbunch_list[i];
			if (self.contains(n)) {
				nodes_list.append(n);
			}
		}
		bunch = nbunch_list.attr("__iter__")();
	}
	return bunch;
}

void _add_one_edge(Graph& self, const py::object& u_of_edge, const py::object& v_of_edge, const py::object& edge_attr) {
	// 1) 端点确保存在并取得 id
	node_t u, v;

	if (self.node_to_id.contains(u_of_edge)) {
		u = py::cast<node_t>(self.node_to_id[u_of_edge]);
	} else {
		// 这里给新节点空属性（与原始代码传 py::none() 等价）
		u = _add_one_node(self, u_of_edge, py::dict{});
	}

	if (self.node_to_id.contains(v_of_edge)) {
		v = py::cast<node_t>(self.node_to_id[v_of_edge]);
	} else {
		v = _add_one_node(self, v_of_edge, py::dict{});
	}

	// 2) 规范化边属性：None -> 空 dict；否则必须是 dict
	py::dict attrs;
	if (edge_attr.is_none()) {
		attrs = py::dict{};
	} else if (py::isinstance<py::dict>(edge_attr)) {
		attrs = edge_attr.cast<py::dict>();
	} else {
		throw py::type_error("edge_attr must be a dict or None.");
	}

	// 3) 初始化双向邻接条目
	self.adj[u][v] = node_attr_dict_factory();
	self.adj[v][u] = node_attr_dict_factory();

	// 4) 填充属性（dict 直接可迭代：kv.first, kv.second）
	for (auto kv : attrs) {
		std::string key = weight_to_string(py::reinterpret_borrow<py::object>(kv.first));
		weight_t    val = py::cast<weight_t>(kv.second);
		self.adj[u][v].insert(std::make_pair(key, val));
		self.adj[v][u].insert(std::make_pair(std::move(key), val));
		// 注意：上面对 key 复用/移动若底层容器需要独立 key，可改为两次构造字符串
	}
}
void Graph_add_edge(Graph& self, const py::object& u_of_edge, const py::object& v_of_edge, const py::kwargs& kwargs) {
    self.dirty_nodes = true;
    self.dirty_adj   = true;

    // 将 **kwargs 复制为独立 dict，避免在 _add_one_edge 内修改到调用方对象
    py::dict edge_attr;
    for (auto kv : kwargs) {
        edge_attr[kv.first] = kv.second;
    }

    _add_one_edge(self, u_of_edge, v_of_edge, edge_attr);  // 其内部负责建点与属性合并
}

void Graph_add_edges(Graph& self, const py::sequence& edges_for_adding, const py::sequence& edges_attr) {
	self.dirty_nodes = true;
	self.dirty_adj   = true;

	const std::size_t n = py::len(edges_for_adding);
	const std::size_t m = py::len(edges_attr);

	if (m != 0 && n != m) {
		// 原代码抛 AssertionError；pybind11 无内建 assertion_error，这里用 value_error 更合适
		throw py::value_error("Edges and Attributes lists must have same length.");
	}

	for (std::size_t i = 0; i < n; ++i) {
		// 1) 取出一条边，要求是 (u, v)
		py::object edge_obj = edges_for_adding[i];
		if (!py::isinstance<py::tuple>(edge_obj)) {
			throw py::type_error("Each edge must be a tuple (u, v).");
		}
		py::tuple e = edge_obj.cast<py::tuple>();
		if (py::len(e) != 2) {
			throw py::type_error("Each edge tuple must have length 2: (u, v).");
		}
		py::object u = e[0];
		py::object v = e[1];

		// 2) 选择该边对应的属性
		py::dict attr;
		if (m != 0) {
			attr = edges_attr[i].cast<py::dict>();  // 若不是 dict 会抛 type_error
		} else {
			attr = py::dict{};
		}

		// 3) 添加该边（_add_one_edge 内部负责必要的建点与属性赋值）
		_add_one_edge(self, u, v, attr);
	}
}

void Graph_add_edges_from(Graph& self, const py::iterable& ebunch, const py::kwargs& attr) {
    self.dirty_nodes = true;
    self.dirty_adj   = true;

    // 1) 复制全局属性 kwargs
    py::dict base_attrs;
    for (auto kv : attr) base_attrs[kv.first] = kv.second;

    // 2) 遍历边集合
    for (py::handle h : ebunch) {
        py::object eobj = py::reinterpret_borrow<py::object>(h);

        // 允许 list/tuple/任意可序列化为 sequence 的对象
        py::sequence e = eobj.cast<py::sequence>();
        const std::size_t L = py::len(e);
        if (L != 2 && L != 3) {
            throw py::value_error(py::str("Edge tuple {} must be a 2-tuple or 3-tuple.").format(eobj));
        }

        py::object u = e[0];
        py::object v = e[1];
        if (u.is_none() || v.is_none()) {
            throw py::value_error("None cannot be a node.");
        }

        // 每边的局部属性（kwargs < per-edge dict）
        py::dict per_edge = py::dict(base_attrs);
        if (L == 3) {
            py::object maybe_dict = e[2];
            if (!py::isinstance<py::dict>(maybe_dict)) {
                throw py::type_error("Edge data (3rd element) must be a dict.");
            }
            py::dict dd = maybe_dict.cast<py::dict>();
            for (auto kv : dd) per_edge[kv.first] = kv.second;
        }

        // 3) 确保端点存在并获得 id
        node_t uid, vid;
        if (self.node_to_id.contains(u)) {
            uid = py::cast<node_t>(self.node_to_id[u]);
        } else {
            uid = _add_one_node(self, u, py::dict{});
        }
        if (self.node_to_id.contains(v)) {
            vid = py::cast<node_t>(self.node_to_id[v]);
        } else {
            vid = _add_one_node(self, v, py::dict{});
        }

        // 4) 初始化（若不存在则创建）边属性容器
        if (!self.adj[uid].count(vid)) self.adj[uid][vid] = node_attr_dict_factory();
        if (!self.adj[vid].count(uid)) self.adj[vid][uid] = node_attr_dict_factory();

        auto& uv = self.adj[uid][vid];
        auto& vu = self.adj[vid][uid];

        // 5) 写入属性：赋值语义（覆盖旧值），两向一致
        for (auto kv : per_edge) {
            std::string k = weight_to_string(py::reinterpret_borrow<py::object>(kv.first));
            weight_t    vval = py::cast<weight_t>(kv.second);
            uv[k] = vval;
            vu[k] = vval;
        }
    }
}

struct commactype : std::ctype<char> {
	commactype() : std::ctype<char>(get_table()) {}

	static const std::ctype_base::mask* get_table() {
		using mask = std::ctype_base::mask;
		static std::vector<mask> table(std::ctype<char>::table_size, mask()); // 静态存储期
		static bool inited = false;
		if (!inited) {
			table[',']  = std::ctype_base::space;
			table[' ']  = std::ctype_base::space;
			table['\t'] = std::ctype_base::space;
			table['\n'] = std::ctype_base::space;
			table['\r'] = std::ctype_base::space;
			inited = true;
		}
		return table.data();
	}
};

void Graph_add_edges_from_file(Graph& self, const py::str& file, bool weighted) {
	self.dirty_nodes = true;
	self.dirty_adj   = true;

	std::ios::sync_with_stdio(false);

	const std::string file_path = py::cast<std::string>(file);
	std::ifstream in(file_path);
	if (!in.is_open()) {
		throw py::value_error("File not found: " + file_path);
	}
	in.imbue(std::locale(std::locale(), new commactype));

	std::string su, sv;
	const std::string key = "weight";
	weight_t w{};

	while (in >> su >> sv) {
		py::str pu(su), pv(sv);

		// 取得/创建端点 id
		node_t uid, vid;
		if (self.node_to_id.contains(pu)) {
			uid = py::cast<node_t>(self.node_to_id[pu]);
		} else {
			uid = _add_one_node(self, pu, py::dict{});
		}
		if (self.node_to_id.contains(pv)) {
			vid = py::cast<node_t>(self.node_to_id[pv]);
		} else {
			vid = _add_one_node(self, pv, py::dict{});
		}

		// 初始化两向邻接条目（若尚不存在）
		if (!self.adj[uid].count(vid)) self.adj[uid][vid] = node_attr_dict_factory();
		if (!self.adj[vid].count(uid)) self.adj[vid][uid] = node_attr_dict_factory();

		if (weighted) {
			if (!(in >> w)) {
				throw py::value_error("Weighted file format error: missing/invalid weight after edge (" + su + ", " + sv + ").");
			}
			self.adj[uid][vid][key] = w;
			self.adj[vid][uid][key] = w;
		}
		// 无权重：仅保证邻接存在，属性留空
	}
}
py::object Graph_add_weighted_edge(Graph& self, py::object u_of_edge, py::object v_of_edge, weight_t weight) {
	self.dirty_nodes = true;
	self.dirty_adj = true;
	py::dict edge_attr;
	edge_attr["weight"] = weight;
	_add_one_edge(self, u_of_edge, v_of_edge, edge_attr);
	return py::object();
}

void Graph_remove_edge(Graph& self, const py::object& u, const py::object& v) {
	self.dirty_nodes = true;
	self.dirty_adj   = true;

	if (!self.node_to_id.contains(u) || !self.node_to_id.contains(v)) {
		throw py::key_error(py::str("No edge {}-{} in graph.").format(u, v));
	}

	const node_t uid = py::cast<node_t>(self.node_to_id[u]);
	const node_t vid = py::cast<node_t>(self.node_to_id[v]);

	// 不创建默认邻接表：用 find 检查
	auto it_u = self.adj.find(uid);
	if (it_u == self.adj.end()) {
		throw py::key_error(py::str("No edge {}-{} in graph.").format(u, v));
	}
	auto& nbrs_u = it_u->second;

	auto it_uv = nbrs_u.find(vid);
	if (it_uv == nbrs_u.end()) {
		throw py::key_error(py::str("No edge {}-{} in graph.").format(u, v));
	}

	// 删除 u->v
	nbrs_u.erase(it_uv);

	// 无向图对称删除 v->u（若自环则不重复）
	if (uid != vid) {
		auto it_v = self.adj.find(vid);
		if (it_v != self.adj.end()) {
			it_v->second.erase(uid);
		}
	}
}
void Graph_remove_edges(Graph& self, const py::sequence& edges_to_remove) {
	self.dirty_nodes = true;
	self.dirty_adj   = true;

	const std::size_t n = py::len(edges_to_remove);
	for (std::size_t i = 0; i < n; ++i) {
		py::object edge_obj = edges_to_remove[i];
		if (!py::isinstance<py::tuple>(edge_obj)) {
			throw py::type_error("Each edge must be a tuple (u, v).");
		}
		py::tuple edge = edge_obj.cast<py::tuple>();
		if (py::len(edge) != 2) {
			throw py::type_error("Each edge tuple must have length 2: (u, v).");
		}
		py::object u = edge[0];
		py::object v = edge[1];

		Graph_remove_edge(self, u, v);
	}
}

int Graph_number_of_edges(const Graph& self, py::object u, py::object v) {
	// 1) 全图边数
	if (u.is_none()) {
		std::size_t total = 0;
		for (const auto& kv : self.adj) total += kv.second.size();
		return static_cast<int>(total / 2);
	}

	// 2) 指定 (u, v) 是否存在：不存在节点 → 0
	if (!self.node_to_id.contains(u) || !self.node_to_id.contains(v)) {
		return 0;
	}

	const node_t uid = py::cast<node_t>(self.node_to_id[u]);
	const node_t vid = py::cast<node_t>(self.node_to_id[v]);

	// 不要用 self.adj[uid] 以免创建空条目
	auto it_u = self.adj.find(uid);
	if (it_u == self.adj.end()) return 0;

	const auto& nbrs_u = it_u->second;
	return static_cast<int>(nbrs_u.count(vid) ? 1 : 0);
}

bool Graph_has_edge(const Graph& self, const py::object& u, const py::object& v) {
	if (!self.node_to_id.contains(u) || !self.node_to_id.contains(v))
		return false;

	const node_t uid = py::cast<node_t>(self.node_to_id[u]);
	const node_t vid = py::cast<node_t>(self.node_to_id[v]);

	auto it_u = self.adj.find(uid);
	if (it_u == self.adj.end())
		return false;

	const auto& nbrs_u = it_u->second;
	return nbrs_u.find(vid) != nbrs_u.end();
}

py::object Graph_copy(py::handle self_h) {
	// 取 C++ 引用
	const Graph& self = py::cast<const Graph&>(self_h);

	// 使用 self.__class__() 构造同类型实例
	py::object cls = py::type::of(self_h);
	py::object Gobj = cls();  // 调用该类型的无参构造；需在绑定里暴露默认构造

	// 取得其 C++ 引用，拷贝字段
	Graph& G = py::cast<Graph&>(Gobj);

	G.graph      = py::dict(self.graph);
	G.id_to_node = py::dict(self.id_to_node);
	G.node_to_id = py::dict(self.node_to_id);
	G.node       = self.node;
	G.adj        = self.adj;
	G.dirty_nodes = self.dirty_nodes;
	G.dirty_adj   = self.dirty_adj;

	return Gobj;  // 返回 Python 对象（保持动态类型）
}

py::dict Graph_degree(const Graph& self, py::object weight) {
	// 1) 预处理权重键
	const bool use_weight = !weight.is_none();
	std::string wkey;
	if (use_weight) {
		wkey = weight_to_string(py::reinterpret_borrow<py::object>(weight));
	}

	// 2) 累加度数：用临时 map，避免反复操作 py::dict
	std::unordered_map<node_t, double> deg;

	// 先为所有节点置 0（保证孤立点也在结果里）
	for (const auto& kv : self.node) {
		deg.emplace(kv.first, 0.0);
	}

	// 3) 遍历边（避免双计：仅在 uid <= vid 时计一次）
	for (const auto& [uid, nbrs] : self.adj) {
		for (const auto& [vid, attrs] : nbrs) {
			if (uid > vid) continue;  // 无向图去重
			double w = 1.0;
			if (use_weight) {
				auto it = attrs.find(wkey);
				if (it != attrs.end()) {
					w = static_cast<double>(it->second);
				}
			}
			deg[uid] += w;
			deg[vid] += w;
		}
	}

	// 4) 生成 Python dict：键为原始 Python 节点对象
	py::dict degree;
	for (const auto& [nid, val] : deg) {
		// self.id_to_node[nid] -> py::object
		py::object node_obj = self.id_to_node.attr("__getitem__")(py::cast(nid));
		degree[node_obj] = val;
	}

	return degree;
}

py::list Graph_neighbors(const Graph& self, const py::object& node) {
	if (!self.node_to_id.contains(node)) {
		throw py::key_error(py::str("No node {}").format(node));
	}

	const node_t uid = py::cast<node_t>(self.node_to_id[node]);

	py::list out;
	auto it = self.adj.find(uid);
	if (it == self.adj.end()) {
		return out; // 没有邻居
	}

	const auto& nbrs = it->second; // map<node_t, ...>
	for (const auto& kv : nbrs) {
		const node_t vid = kv.first;
		// 从 id_to_node 取回 Python 节点对象（const 上用 __getitem__ 避免 operator[]）
		py::object v = self.id_to_node.attr("__getitem__")(py::cast(vid));
		out.append(v);
	}
	return out;
}

py::object Graph_nodes_subgraph(py::handle self_h, const py::sequence& from_nodes) {
	const Graph& self = py::cast<const Graph&>(self_h);

	// 1) 构造“同类”实例：等价于 Python 的 self.__class__()
	py::object cls  =  py::type::of(self_h);
	py::object Gobj = cls();                       // 需要 Graph 在绑定里有无参构造
	Graph& G        = py::cast<Graph&>(Gobj);

	// 2) 复制元数据 graph（浅拷贝）
	G.graph = py::dict(self.graph);

	// 3) 收集子集节点（只加入存在于原图的节点）
	std::unordered_set<node_t> subset;
	const std::size_t n = py::len(from_nodes);
	for (std::size_t i = 0; i < n; ++i) {
		py::object node = from_nodes[i];
		if (!self.node_to_id.contains(node)) continue;

		node_t nid = py::cast<node_t>(self.node_to_id[node]);
		subset.insert(nid);

		// 取该节点属性 -> py::dict
		py::dict node_attr;
		auto it = self.node.find(nid);
		if (it != self.node.end()) {
			for (const auto& kv : it->second) {
				node_attr[py::str(kv.first)] = kv.second;  // weight_t 可隐式/显式 cast
			}
		}

		// 在子图中建点（保持原 Python 节点对象）
		_add_one_node(G, node, node_attr);
	}

	// 4) 子集诱导边：仅当两端都在 subset 中；无向图用 uid <= vid 去重
	for (node_t uid : subset) {
		auto it_u = self.adj.find(uid);
		if (it_u == self.adj.end()) continue;

		for (const auto& kv : it_u->second) {
			node_t vid = kv.first;
			if (!subset.count(vid)) continue;
			if (uid > vid) continue;  // 无向图避免重复添加

			// 边属性 -> py::dict
			py::dict edge_attr;
			for (const auto& e : kv.second) {
				edge_attr[py::str(e.first)] = e.second;
			}

			// 从 id 取回 Python 节点对象
			py::object u_obj = self.id_to_node.attr("__getitem__")(py::cast(uid));
			py::object v_obj = self.id_to_node.attr("__getitem__")(py::cast(vid));

			_add_one_edge(G, u_obj, v_obj, edge_attr);
		}
	}

	return Gobj;
}
py::object Graph_ego_subgraph(py::object self, py::object center) {
	py::list neighbors_of_center = py::list(self.attr("all_neighbors")(center));
	neighbors_of_center.append(center);
	return self.attr("nodes_subgraph")(neighbors_of_center);
}

py::object Graph_size(const Graph& G, py::object weight ) {
	py::dict deg = Graph_degree(G, weight);  // {node_obj: degree_value}

	double sum_deg = 0.0;
	for (auto kv : deg) {
		sum_deg += py::cast<double>(kv.second);
	}

	if (weight.is_none()) {
		// 与原逻辑一致：先转整型再除 2（整数除法）
		// 这里 sum_deg 应该是偶数（无向图），做一次显式截断
		long long m2 = static_cast<long long>(sum_deg);
		return py::int_(m2 / 2);
	} else {
		return py::float_(sum_deg / 2.0);
	}
}

bool Graph_is_directed(const Graph& self) {
	return false;
}

bool Graph_is_multigraph(const Graph& self) {
	return false;
}

py::object Graph::get_nodes() {
	py::object MappingProxyType = py::module_::import("types").attr("MappingProxyType");
	if (this->dirty_nodes) {
		py::dict nodes = py::dict();
		for (const auto& node_info : node) {
			node_t id = node_info.first;
			const auto& node_attr = node_info.second;
			nodes[this->id_to_node[py::int_(id)]] = MappingProxyType(attr_to_dict(node_attr));
		}
		this->nodes_cache = MappingProxyType(nodes);
		this->dirty_nodes = false;
	}
	return this->nodes_cache;
}

py::object Graph::get_name() {
	return this->graph.attr("get")("name", "");
}

py::object Graph::get_graph() {
	return this->graph;
}

py::object Graph::get_adj() {
	py::object MappingProxyType = py::module_::import("types").attr("MappingProxyType");
	if (this->dirty_adj) {
		py::dict adj = py::dict();
		for (const auto& ego_edges : this->adj) {
			node_t start_point = ego_edges.first;
			py::dict ego_edges_dict = py::dict();
			for (const auto& edge_info : ego_edges.second) {
				node_t end_point = edge_info.first;
				const auto& edge_attr = edge_info.second;
				ego_edges_dict[this->id_to_node[py::int_(end_point)]] = MappingProxyType(attr_to_dict(edge_attr));
			}
			adj[this->id_to_node[py::int_(start_point)]] = MappingProxyType(ego_edges_dict);
		}
		this->adj_cache = MappingProxyType(adj);
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
			if (seen.find(std::make_pair(u,v)) == seen.end()) {
				seen.insert(std::make_pair(u,v));
				seen.insert(std::make_pair(v,u));
				edges.append(py::make_tuple(this->id_to_node[py::int_(u)], this->id_to_node[py::int_(v)], attr_to_dict(edge_attr)));
			}
		}
	}
	return edges;
}
