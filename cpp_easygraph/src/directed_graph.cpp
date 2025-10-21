#include "directed_graph.h"
#include "common.h"
#include "utils.h"

DiGraph::DiGraph() : Graph() {}

void DiGraph___init__(DiGraph &self, py::kwargs kwargs) {
  new (&self) DiGraph();

  py::object MappingProxyType =
      py::module_::import("types").attr("MappingProxyType");

  if (kwargs) {
    for (auto &item : kwargs) {
      self.graph[py::str(item.first)] = item.second;
    }
  }

  self.nodes_cache = MappingProxyType(py::dict());
  self.adj_cache = MappingProxyType(py::dict());
}

py::dict DiGraph_out_degree(const py::object &self,
                                   const py::object &weight) {
  py::dict degree;

  // edges 既可能是属性(list/iterable)，也可能是方法（如 edges(data=True)）
  py::object edges_obj = self.attr("edges");

  py::list edges = py::isinstance<py::function>(edges_obj)
                       ? py::list(edges_obj())
                       : py::list(edges_obj);

  // 遍历边：(u, v, d)
  for (py::ssize_t i = 0; i < py::len(edges); ++i) {
    py::tuple edge = edges[i].cast<py::tuple>();
    py::object u = edge[0];
    // v 在本函数中未使用，但保留一致性
    // py::object v = edge[1];
    py::dict d = (edge.size() >= 3) ? edge[2].cast<py::dict>() : py::dict();

    py::object w = d.attr("get")(weight, 1); // d.get(weight, 1)

    if (degree.contains(u)) {
      // degree[u] += w  —— 用 Python 的 __add__ 避免数值类型分歧
      degree[u] = degree[u].attr("__add__")(w);
    } else {
      degree[u] = w;
    }
  }

  // 确保所有节点都有条目：不存在则补 0
  py::object nodes_obj = self.attr("nodes");
  py::list nodes = py::isinstance<py::function>(nodes_obj)
                       ? py::list(nodes_obj())
                       : py::list(nodes_obj);
  for (py::ssize_t i = 0; i < py::len(nodes); ++i) {
    py::object node = nodes[i];
    if (!degree.contains(node)) {
      degree[node] = 0;
    }
  }

  return degree;
}

py::dict DiGraph_in_degree(const py::object &self,
                                  const py::object &weight) {
  py::dict degree;

  // edges 可能是属性(list/iterable)或方法（如 edges(data=True)）
  py::object edges_obj = self.attr("edges");
  py::list edges = py::isinstance<py::function>(edges_obj)
                       ? py::list(edges_obj())
                       : py::list(edges_obj);

  // 遍历边：(u, v, d)
  for (py::ssize_t i = 0; i < py::len(edges); ++i) {
    py::tuple edge = edges[i].cast<py::tuple>();
    // u 未使用，但保留对齐
    // py::object u = edge[0];
    py::object v = edge[1];
    py::dict d = (edge.size() >= 3) ? edge[2].cast<py::dict>() : py::dict();

    // d.get(weight, 1)
    py::object w = d.attr("get")(weight, 1);

    if (degree.contains(v)) {
      // degree[v] += w   —— 用 Python 的 __add__ 以避免数值类型差异
      degree[v] = degree[v].attr("__add__")(w);
    } else {
      degree[v] = w;
    }
  }

  // 确保所有节点都有条目：不存在则补 0
  py::object nodes_obj = self.attr("nodes");
  py::list nodes = py::isinstance<py::function>(nodes_obj)
                       ? py::list(nodes_obj())
                       : py::list(nodes_obj);
  for (py::ssize_t i = 0; i < py::len(nodes); ++i) {
    py::object node = nodes[i];
    if (!degree.contains(node)) {
      degree[node] = 0;
    }
  }

  return degree;
}

py::dict DiGraph_degree(const py::object &self,
                               const py::object &weight) {
  py::dict degree;

  // out_degree / in_degree 都应返回 dict
  py::dict out_degree = self.attr("out_degree")(weight).cast<py::dict>();
  py::dict in_degree = self.attr("in_degree")(weight).cast<py::dict>();

  // nodes 可能是属性或方法
  py::object nodes_obj = self.attr("nodes");

  py::list nodes = py::isinstance<py::function>(nodes_obj)
                       ? py::list(nodes_obj())
                       : py::list(nodes_obj);

  for (py::ssize_t i = 0; i < py::len(nodes); ++i) {
    py::object u = nodes[i];

    // 取度值：若不存在则当 0
    py::object out_v = out_degree.contains(u) ? out_degree[u] : py::int_(0);
    py::object in_v = in_degree.contains(u) ? in_degree[u] : py::int_(0);

    // degree[u] = out_v + in_v  （用 Python 的 __add__ 以兼容 int/float/Decimal
    // 等）
    degree[u] = out_v.attr("__add__")(in_v);
  }
  return degree;
}

py::object DiGraph_size(const py::object &self,
                               const py::object &weight) {
  py::dict out_degree = self.attr("out_degree")(weight).cast<py::dict>();

  py::object values = out_degree.attr("values")(); // dict_values 视图
  py::object s = py::module_::import("builtins").attr("sum")(values);

  return weight.is_none() ? py::int_(s) : s;
}

py::object DiGraph_number_of_edges(const py::object &self,
                                          const py::object &u,
                                          const py::object &v) {
  // 若未指定 u（与原先 u == py::object() 等价），直接返回 size()
  if (u.is_none()) {
    return self.attr("size")();
  }

  // 将 self 视为 Graph 的引用
  Graph &G = self.cast<Graph &>();

  // node_to_id: Python dict（与原代码保持一致）
  py::dict node_to_id = G.node_to_id;

  // u_id = node_to_id[u]
  node_t u_id = node_to_id[u].cast<node_t>();

  // v_id = node_to_id.get(v, -1)
  node_t v_id = node_to_id.attr("get")(v, py::int_(-1)).cast<node_t>();

  // 计算是否存在边 (u, v)
  int exists =
      (v_id != static_cast<node_t>(-1)) && (G.adj[u_id].count(v_id) ? 1 : 0);
  return py::int_(exists); // 与原实现保持返回 0/1 的 int
}
