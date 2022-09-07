import os

from numbers import Number
from typing import Iterator

import easygraph as eg


def fuzzy_equal(o1, o2):
    if isinstance(o1, dict) and isinstance(o2, dict):
        if set(o1.keys()) != set(o2.keys()):
            return False
        for key in o1.keys():
            o1_, o2_ = o1[key], o2[key]
            if not fuzzy_equal(o1_, o2_):
                return False
        return True
    if isinstance(o1, Iterator) and isinstance(o2, Iterator):
        return fuzzy_equal(list(o1), list(o2))
    if isinstance(o1, list) and isinstance(
        o2, list
    ):  # every item in list1 should be in list2
        if len(o1) != len(o2):
            return False
        for item1 in o1:
            belong = False
            for item2 in o2:
                if fuzzy_equal(item1, item2):
                    belong = True
                    break
            if not belong:
                print(item1)
                return False
        return True
    if isinstance(o1, tuple) and isinstance(
        o2, tuple
    ):  # corresponding items should be equal
        if len(o1) != len(o2):
            return False
        for i in range(len(o1)):
            if not fuzzy_equal(o1[i], o2[i]):
                return False
        return True
    if isinstance(o1, Number) and isinstance(o2, Number):
        return abs(o2 - o1) < 1e-6
    return o1 == o2


class Tester:
    def __init__(self, class1, class2):
        self.class1 = class1
        self.class2 = class2

    def run_method(self, name, *args, **kwargs):
        r1 = getattr(self.G1, name)(*args, **kwargs)
        r2 = getattr(self.G2, name)(*args, **kwargs)
        return r1, r2

    def assert_object(self, o1, o2):
        if not fuzzy_equal(o1, o2):
            print(f"o1: {o1}")
            print(f"o2: {o2}")
            raise AssertionError(
                f"FAILED: o1 != o2 in test for {self.class1.__name__} and"
                f" {self.class2.__name__}"
            )

    def assert_property(self, name, g1=None, g2=None):
        if g1 is None:
            g1 = self.G1
        if g2 is None:
            g2 = self.G2
        r1 = getattr(g1, name)
        r2 = getattr(g2, name)
        if name == "edges":
            if not g1.is_directed():
                r1 = r1 + [
                    (edge[1], edge[0])
                    if len(edge) == 2
                    else (edge[1], edge[0], edge[2])
                    for edge in r1
                ]
                r2 = r2 + [
                    (edge[1], edge[0])
                    if len(edge) == 2
                    else (edge[1], edge[0], edge[2])
                    for edge in r2
                ]
        self.assert_object(r1, r2)

    def assert_method(self, name, *args, **kwargs):
        r1 = getattr(self.G1, name)(*args, **kwargs)
        r2 = getattr(self.G2, name)(*args, **kwargs)
        self.assert_object(r1, r2)

    def assert_graph(self, g1, g2):
        self.assert_property("graph", g1, g2)
        self.assert_property("nodes", g1, g2)
        self.assert_property("edges", g1, g2)
        self.assert_property("adj", g1, g2)

    def test(self):
        self.G1 = self.class1(name="graph", time=0)
        self.G2 = self.class2(name="graph", time=0)
        self.assert_property("graph")

        self.run_method("add_node", 1, x=2)
        self.assert_property("nodes")

        self.run_method("add_nodes", [2])
        self.run_method("add_nodes", [3, 4], [{"x": 2}, {"x": 2}])
        self.assert_property("nodes")

        self.run_method("add_nodes_from", [5], y=3)
        self.assert_property("nodes")

        self.assert_property("adj")

        self.run_method("add_edges", [(1, 2)])
        self.run_method("add_edges", [(2, 3), (1, 3)], [{"weight": 1}, {"weight": 1}])
        self.assert_property("edges")

        self.run_method("add_edges_from", [(1, 4)], we=2)
        self.assert_property("edges")

        self.assert_property("adj")

        with open("test.txt", "w") as f:
            f.writelines(["6,7\n"])
        self.run_method("add_edges_from_file", "test.txt")
        with open("test.txt", "w") as f:
            f.writelines(["8,9,10\n", "9,10,11\n"])
        self.run_method("add_edges_from_file", "test.txt", True)
        os.remove("test.txt")
        self.assert_property("edges")

        self.run_method("remove_node", "8")
        self.run_method("remove_nodes", ["9", "10"])
        self.assert_property("nodes")

        self.assert_property("edges")
        self.assert_property("adj")

        self.run_method("remove_edge", "6", "7")
        self.run_method("remove_edges", [(2, 3), (1, 3)])
        self.assert_property("edges")

        self.assert_property("adj")

        self.assert_method("has_node", 1)
        self.assert_method("has_node", 10)
        self.assert_method("has_edge", 1, 2)
        self.assert_method("has_edge", 2, 3)
        self.assert_method("number_of_edges")
        self.assert_method("number_of_edges", 2)
        self.assert_method("number_of_nodes")
        self.assert_method("is_directed")
        self.assert_method("is_multigraph")
        self.assert_method("degree", "we")
        self.assert_method("size")
        self.assert_method("size", "we")

        if self.G1.is_directed():
            self.assert_method("in_degree", "we")
            self.assert_method("out_degree")

        G_1, G_2 = self.run_method("copy")
        G_1.add_edge(-1, -1)
        G_2.add_edge(-1, -1)
        self.assert_graph(G_1, G_2)
        self.assert_graph(self.G1, self.G2)
        self.assert_object(G_1.has_edge(-1, -1), True)
        self.assert_object(self.G1.has_edge(-1, -1), False)

        G_2 = self.G2.py()
        self.assert_graph(self.G1, G_2)
        G_1 = self.G1.cpp()
        self.assert_graph(G_1, self.G2)

        G_1, G_2 = self.run_method("nodes_subgraph", [1, 2, 3, "6"])
        self.assert_graph(G_1, G_2)

        G_1, G_2 = self.run_method("ego_subgraph", 1)
        self.assert_graph(G_1, G_2)

        (G_1, _, node_of_index_1), (
            G_2,
            _,
            node_of_index_2,
        ) = self.run_method("to_index_node_graph")
        G_1_nodes = {node_of_index_1[i]: j for i, j in G_1.nodes.items()}
        G_1_adj = {
            node_of_index_1[i]: {node_of_index_1[a]: b for a, b in j.items()}
            for i, j in G_1.adj.items()
        }
        G_2_nodes = {node_of_index_2[i]: j for i, j in G_2.nodes.items()}
        G_2_adj = {
            node_of_index_2[i]: {node_of_index_2[a]: b for a, b in j.items()}
            for i, j in G_2.adj.items()
        }
        self.assert_object(G_1_nodes, G_2_nodes)
        self.assert_object(G_1_adj, G_2_adj)

        self.assert_method("__len__")

        self.assert_method("__contains__", 1)
        self.assert_method("__contains__", 10)

        self.assert_method("__getitem__", 1)

        self.assert_method("__iter__")

        print(f"PASSED: Test for {self.class1.__name__} and {self.class2.__name__}")


if __name__ == "__main__":
    graph_tester = Tester(eg.Graph, eg.GraphC)
    digraph_tester = Tester(eg.DiGraph, eg.DiGraphC)

    graph_tester.test()
    digraph_tester.test()
