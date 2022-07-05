#include "ModuleMethods.h"

std::unordered_map<long long, float> sum_nmw_rec, max_nmw_rec, local_constraint_rec;

float mutual_weight(mimimsf_t& G, int u, int v, std::string weight = "None") {
	float a_uv = 0, a_vu = 0;
	if (G.count(u) && G[u].count(v)) {
		msf_t& guv = G[u][v];
		a_uv = guv.count(weight) ? guv[weight] : 1;
	}
	if (G.count(v) && G[v].count(u)) {
		msf_t& gvu = G[v][u];
		a_uv = gvu.count(weight) ? gvu[weight] : 1;
	}
	return a_uv + a_vu;
}

float normalized_mutual_weight(mimimsf_t& G, int u, int v, std::string weight = "None", int norm = 0) {
	edge_tuple edge = { u, v };
	auto& nmw_rec = norm ? sum_nmw_rec : max_nmw_rec;
	if (nmw_rec.count(edge.val)) {
		return nmw_rec[edge.val];
	}
	else {
		float scale = 0;
		for (auto& w : G[u]) {
			scale += mutual_weight(G, u, w.first, weight);
		}
		float nmw;
		nmw = scale ? mutual_weight(G, u, v, weight) / scale : 0;
		nmw_rec[edge.val] = nmw;
		return nmw;
	}
}

inline float redundancy(mimimsf_t& G, int u, int v, std::string weight = "None") {
	float r = 0;
	for (auto& w : G[u]) {
		r += normalized_mutual_weight(G, u, w.first, weight) * normalized_mutual_weight(G, v, w.first, weight);
	}
	return 1 - r;
}

PyObject* effective_size(PyObject* easygraph, PyObject* args, PyObject* kwargs) {
	PyObject* nodes = Py_None, * weight = Py_None;
	Graph *graph = nullptr;
	static char* kwlist[] = { (char*)"G", (char*)"nodes", (char*)"weight", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", kwlist, &graph, &nodes, &weight))
		return nullptr;
	
	sum_nmw_rec.clear();
	max_nmw_rec.clear();
	PyObject* ret = PyDict_New();
	mimimsf_t& G = graph->adj;
	if (PyObject_CallMethod((PyObject*)graph, "is_directed", "()") == Py_False && weight == Py_None) {
		if (nodes == Py_None) {
			for (auto& v : G) {
				PyObject* pv = PyLong_FromLong(v.first);
				if (!v.second.size()) {
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, pv), PyFloat_FromDouble(Py_NAN));
					continue;
				}
				Graph* E = (Graph*)PyObject_CallMethod((PyObject*)graph, "ego_subgraph", "(O)", PyDict_GetItem(graph->id_to_node, pv));
				if (E->node.size() > 1) {
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, pv), PyFloat_FromDouble(E->node.size() - 1 - 1.0 * (2 * PyLong_AsLong(PyObject_CallMethod((PyObject*)E, "size", "()"))) / (E->node.size() - 1)));
				}
				else {
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, pv), PyFloat_FromDouble(0));
				}
			}
		}
		else {
			for (Py_ssize_t i = 0; i < PyList_Size(nodes);i++) {
				PyObject* pv = PyDict_GetItem(graph->node_to_id, PyList_GetItem(nodes, i));
				int v = PyLong_AsLong(pv);
				if (!G[v].size()) {
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, pv), PyFloat_FromDouble(Py_NAN));
					continue;
				}
				Graph* E = (Graph*)PyObject_CallMethod((PyObject*)graph, "ego_subgraph", "(O)", PyDict_GetItem(graph->id_to_node, pv));
				if (E->adj.size() > 1) {
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, pv), PyFloat_FromDouble(E->node.size() - 1 - 1.0 * (2 * PyLong_AsLong(PyObject_CallMethod((PyObject*)E, "size", "()"))) / (E->node.size() - 1)));
				}
				else {
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, pv), PyFloat_FromDouble(0));
				}
			}
		}
	}
	else {
		if (nodes == Py_None) {
			for (auto& v : G) {
				PyObject* pv = PyLong_FromLong(v.first);
				if (!v.second.size()) {
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, pv), PyFloat_FromDouble(Py_NAN));
					continue;
				}
				else{
					float sum = 0;
					for (auto& u : v.second) {
						sum += redundancy(G, v.first, u.first, weight == Py_None ? std::string("None") : std::string(PyUnicode_AsUTF8(weight)));
					}
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, pv), PyFloat_FromDouble(sum));
				}
			}
		}
		else {
			for (Py_ssize_t i = 0; i < PyList_Size(nodes);i++) {
				int v = PyLong_AsLong(PyDict_GetItem(graph->node_to_id, PyList_GetItem(nodes, i)));
				if (!G[v].size()) {
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, PyLong_FromLong(v)), PyFloat_FromDouble(Py_NAN));
					continue;
				}
				else {
					float sum = 0;
					for (auto& u : G[v]) {
						sum += redundancy(G, v, u.first, weight == Py_None ? std::string("None") : std::string(PyUnicode_AsUTF8(weight)));
					}
					PyDict_SetItem(ret, PyDict_GetItem(graph->id_to_node, PyLong_FromLong(v)), PyFloat_FromDouble(sum));
				}
			}
		}
	}
	return ret;
}

float local_constraint(mimimsf_t& G, int u, int v, std::string weight = "None") {
	edge_tuple edge = { u, v };
	if (local_constraint_rec.count(edge.val)) {
		return local_constraint_rec[edge.val];
	}
	else {
		float direct = normalized_mutual_weight(G, u, v, weight);
		float indirect = 0;
		for (auto& w : G[u]) {
			indirect += normalized_mutual_weight(G, u, w.first, weight) * normalized_mutual_weight(G, w.first, v, weight);
		}
		float result = (direct + indirect) * (direct + indirect);
		local_constraint_rec[edge.val] = result;
		return result;
	}
}

std::pair<int, double> compute_constraint_of_v(mimimsf_t& G, int v, std::string weight) {
	double constraint_of_v = 0;
	if (G[v].size() == 0) {
		constraint_of_v = Py_NAN;
	}
	else {
		for (auto& n : G[v]) {
			constraint_of_v += local_constraint(G, v, n.first, weight);
		}
	}
	return std::make_pair(v, constraint_of_v);
}

PyObject* constraint(PyObject* easygraph, PyObject* args, PyObject* kwargs) {
	Graph* graph;
	PyObject* nodes = Py_None, * weight = Py_None, * n_workers = Py_None;
	static char* kwlist[] = { (char*)"G", (char*)"nodes", (char*)"weight", (char*)"n_workers", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOO", kwlist, &graph, &nodes, &weight, &n_workers))
		return nullptr;
	sum_nmw_rec.clear();
	max_nmw_rec.clear();
	local_constraint_rec.clear();
	std::vector<std::pair<int, double>> constraints_results;
	mimimsf_t& G = graph->adj;
	if (n_workers == Py_None) {
		if (nodes == Py_None) {
			for (auto& v : graph->node) {
				constraints_results.push_back(compute_constraint_of_v(G, v.first, weight == Py_None ? std::string("None") : std::string(PyUnicode_AsUTF8(weight))));
			}
		}
		else {
			for (Py_ssize_t i = 0;i < PyList_Size(nodes);i++) {
				int v = PyLong_AsLong(PyDict_GetItem(graph->node_to_id, PyList_GetItem(nodes, i)));
				constraints_results.push_back(compute_constraint_of_v(G, v, weight == Py_None ? std::string("None") : std::string(PyUnicode_AsUTF8(weight))));
			}
		}
	}
	PyObject* constraint = PyDict_New();
	for (auto& each : constraints_results) {
		PyDict_SetItem(constraint, PyDict_GetItem(graph->id_to_node, PyLong_FromLong(each.first)), PyFloat_FromDouble(each.second));
	}
	return constraint;
}

PyObject* hierarchy(PyObject* easygraph, PyObject* args, PyObject* kwargs) {
	Graph* graph;
	PyObject* nodes = Py_None, * weight = Py_None;
	PyObject* hierarchy = PyDict_New();
	static char* kwlist[] = { (char*)"G", (char*)"nodes", (char*)"weight", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", kwlist, &graph, &nodes, &weight))
		return nullptr;
	mimimsf_t& G = graph->adj;
	PyObject* con = PyObject_CallMethod(easygraph, "cpp_constraint", "(O)", graph);
	if (nodes == Py_None) {
		for (auto& v : graph->node) {
			Graph* E = (Graph*)PyObject_CallMethod((PyObject*)graph, "ego_subgraph", "(O)", PyDict_GetItem(graph->id_to_node, PyLong_FromLong(v.first)));
			int n = E->node.size() - 1;
			float C = 0;
			std::unordered_map<int, float> c;
			for (auto& w : G[v.first]) {
				C += local_constraint(G, v.first, w.first);
				c[w.first] = local_constraint(G, v.first, w.first);
			}
			if (n > 1) {
				float value = 0;
				for (auto w : G[v.first]) {
					value += c[w.first] / C * n * log(c[w.first] / C * n) / (n * log(n));
				}
				PyDict_SetItem(hierarchy, PyDict_GetItem(graph->id_to_node, PyLong_FromLong(v.first)), PyFloat_FromDouble(value));
			}
			Py_DecRef((PyObject*)E);
		}
	}
	else {
		for (Py_ssize_t i = 0;i < PyList_Size(nodes);i++) {
			int v = PyLong_AsLong(PyDict_GetItem(graph->node_to_id, PyList_GetItem(nodes, i)));
			Graph* E = (Graph*)PyObject_CallMethod((PyObject*)graph, "ego_subgraph", "(O)", PyDict_GetItem(graph->id_to_node, PyLong_FromLong(v)));
			int n = E->node.size() - 1;
			float C = 0;
			std::unordered_map<int, float> c;
			for (auto& w : G[v]) {
				C += local_constraint(G, v, w.first);
				c[w.first] = local_constraint(G, v, w.first);
			}
			if (n > 1) {
				float value = 0;
				for (auto w : G[v]) {
					value += c[w.first] / C * n * log(c[w.first] / C * n) / (n * log(n));
				}
				PyDict_SetItem(hierarchy, PyDict_GetItem(graph->id_to_node, PyLong_FromLong(v)), PyFloat_FromDouble(value));
			}
			Py_DecRef((PyObject*)E);
		}
	}
	return hierarchy;
}

struct dijkstra_node {
	float dist;
	int node;
};
bool operator < (const dijkstra_node& u, const dijkstra_node& v) {
	return u.dist == v.dist ? u < v : u.dist < v.dist;
}

PyObject* _dijkstra_multisource(PyObject* easygraph, PyObject* args, PyObject* kwargs) {
	Graph* G;
	PyObject* sources;
	PyObject* pweight = PyUnicode_FromString("weight"), * target = Py_None;
	static char* kwlist[] = { (char*)"G", (char*)"sources", (char*)"weight", (char*)"target", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &G, &sources, &pweight, &target))
		return nullptr;
	int targetid = -1;
	if (PyDict_Contains(G->node_to_id, target))
		targetid = PyLong_AsLong(PyDict_GetItem(G->node_to_id, target));
	std::string weight(PyUnicode_AsUTF8(pweight));
	std::priority_queue<dijkstra_node> qu;
	std::map<int, float> dist, seen;
	PyObject* iterator = PyObject_GetIter(sources);
	if (iterator == nullptr) {
		// ...
		return nullptr;
	}
	PyObject* source;
	while ((source = PyIter_Next(iterator))) {
		int sourceid = -1;
		if (PyDict_Contains(G->node_to_id, source) == 1) {
			sourceid = PyLong_AsLong(PyDict_GetItem(G->node_to_id, source));
			seen[sourceid] = 0;
			qu.push({ 0, sourceid });
		}
		else {
			//...
			return nullptr;
		}
		Py_DecRef(source);
	}
	Py_DecRef(iterator);
	while (!qu.empty()) {
		dijkstra_node n = qu.top();
		int v = n.node;
		qu.pop();
		if (dist.find(v) != dist.end())
			continue;
		dist[v] = n.dist;
		if (n.node == targetid) {
			break;
		}
		for (auto& p : G->adj[v]) {
			int u = p.first;
			auto& weights = G->adj[v][u];
			float cost = weights.find(weight) != weights.end() ? weights[weight] : 1;
			float vu_dist = dist[v] + cost;
			if (dist.find(u) != dist.end()) {
				if (vu_dist < dist[u]) {
					PyErr_Format(PyExc_ValueError, "Contradictory paths found: negative weights?");
					return nullptr;
				}
			}
			else if (seen.find(u) == seen.end() || vu_dist < seen[u]) {
				seen[u] = vu_dist;
				dijkstra_node dn = { vu_dist, u };
				qu.push(dn);
			}
			else
				continue;
		}
	}
	PyObject* pdist = PyDict_New();
	for (auto& p : dist)
		PyDict_SetItem(pdist, PyDict_GetItem(G->id_to_node, PyLong_FromLong(p.first)), PyFloat_FromDouble(p.second));
	return pdist;
}


PyObject* _biconnected_dfs_record_edges(PyObject* easygraph, PyObject* args, PyObject* kwargs) {
	Graph* G;
	PyObject* need_components = Py_True;
	static char* kwlist[] = { (char*)"G", (char*)"need_components", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &G, &need_components))
		return nullptr;
	std::set<int> visited;
	PyObject* ret = PyList_New(0);
	struct edge {
		int u, v;
	};

	struct stack_node {
		int parent, child;
		mimsf_it iter, end;
	};

	for (auto& p : G->node) {
		int start = p.first;
		if (visited.find(start) != visited.end())
			continue;
		std::map<int, int> discovery, low;
		discovery.insert(std::make_pair(start, 0));
		low.insert(std::make_pair(start, 0));
		int root_children = 0;
		visited.insert(start);
		std::vector<edge> edge_stack;
		std::vector<stack_node> stack;
		stack.push_back({ start, start,G->adj[start].begin(),G->adj[start].end()});
		while (!stack.empty()) {
			stack_node &node = stack.back();
			int grandparent = node.parent, parent = node.child;
			if (node.iter != node.end) {
				int child = node.iter->first;
				if (grandparent == child) {
					node.iter++;
					continue;
				}
				if (visited.find(child) != visited.end()) {
					if (discovery[child] <= discovery[parent]) {
						low[parent] = std::min(low[parent], discovery[child]);
						if (PyObject_IsTrue(need_components)) {
							edge_stack.push_back({ parent, child });
						}
					}
				}
				else {
					low[child] = discovery[child] = discovery.size();
					visited.insert(child);
					stack.push_back({ parent, child, G->adj[child].begin(),G->adj[child].end() });
					if (PyObject_IsTrue(need_components)) {
						edge_stack.push_back({ parent, child });
					}
				}
				node.iter++;
			}
			else { //迭代终止
				stack.pop_back();
				if (stack.size() > 1) {
					if (low[parent] >= discovery[grandparent]) {
						if (PyObject_IsTrue(need_components)) {
							edge e;
							PyObject* temp_ret = PyList_New(0);
							do {
								e = edge_stack.back();
								edge_stack.pop_back();
								PyList_Append(temp_ret, Py_BuildValue("(OO)",
									PyDict_GetItem(G->id_to_node, PyLong_FromLong(e.u)),
									PyDict_GetItem(G->id_to_node, PyLong_FromLong(e.v))));
							} while (e.u != grandparent || e.v != parent);
							PyList_Append(ret, temp_ret);
						}
						else {
							PyList_Append(ret, PyDict_GetItem(G->id_to_node, PyLong_FromLong(grandparent)));
						}
					}
				}
				else if (stack.size()) {
					root_children += 1;
					if (PyObject_IsTrue(need_components)) {
						edge e;
						PyObject* temp_ret = PyList_New(0);
						auto iter = edge_stack.rbegin();
						do {
							e = *iter;
							PyList_Append(temp_ret, Py_BuildValue("(OO)",
								PyDict_GetItem(G->id_to_node, PyLong_FromLong(e.u)),
								PyDict_GetItem(G->id_to_node, PyLong_FromLong(e.v))));
							iter++;
						} while (e.u != grandparent || e.v != parent);
						PyList_Append(ret, temp_ret);
					}
				}
			}
		}
		if (!PyObject_IsTrue(need_components)) {
			if (root_children > 1) {
				PyList_Append(ret, PyDict_GetItem(G->id_to_node, PyLong_FromLong(start)));
			}
		}
	}
	return ret;
}
