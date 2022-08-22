#!/usr/bin/env python3

"""
Yeah, I tried for hours but I can't get it to work with pytest.
Just write in pytest next time. :)

Teddy
"""

import os

from typing import Iterator

import easygraph as eg

from pytest import approx


def fuzzy_equal(o1, o2):
    if isinstance(o1, dict) and isinstance(o2, dict):
        if set(o1.keys()) != set(o2.keys()):
            return False
        for key in o1.keys():
            o1_, o2_ = o1[key], o2[key]
            if not fuzzy_equal(o1_, o2_):
                return False
        return True
    if str(o1).isdigit() and str(o2).isdigit():
        # return abs(o2 - o1) < 1e-6
        assert o1 == approx(o2)
    if isinstance(o1, Iterator) and isinstance(o2, Iterator):
        return fuzzy_equal(list(o1), list(o2))
    return o1 == o2


class CPPGraphTestBase:
    # def __init__(self, class1, class2):
    @classmethod
    def setup_method(cls):
        cls.class1 = object
        cls.class2 = object
        cls.G1 = cls.class1(name="graph", time=0)  # type: ignore
        cls.G2 = cls.class2(name="graph", time=0)  # type: ignore

    @classmethod
    def run_method(cls, name, *args, **kwargs):
        r1 = getattr(cls.G1, name)(*args, **kwargs)
        r2 = getattr(cls.G2, name)(*args, **kwargs)
        return r1, r2

    @classmethod
    def assert_object(cls, o1, o2):
        if not fuzzy_equal(o1, o2):
            print(f"o1: {o1}")
            print(f"o2: {o2}")
            raise AssertionError(
                f"FAILED: o1 != o2 in test for {cls.class1.__name__} and"
                f" {cls.class2.__name__}"
            )

    @classmethod
    def assert_property(cls, name):
        r1 = getattr(cls.G1, name)
        r2 = getattr(cls.G2, name)
        cls.assert_object(r1, r2)

    def assert_method(self, name, *args, **kwargs):
        r1 = getattr(self.G1, name)(*args, **kwargs)
        r2 = getattr(self.G2, name)(*args, **kwargs)
        self.assert_object(r1, r2)

    def assert_graph(self, g1, g2):
        self.assert_object(g1.graph, g2.graph)
        self.assert_object(g1.nodes, g2.nodes)
        self.assert_object(g1.edges, g2.edges)
        self.assert_object(g1.adj, g2.adj)

    def test_graph(self):
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

        if self.G1.is_directed():  # type: ignore
            self.assert_method("in_degree", "we")
            self.assert_method("out_degree")

        G_1, G_2 = self.run_method("copy")
        G_1.add_edge(-1, -1)
        G_2.add_edge(-1, -1)
        self.assert_graph(G_1, G_2)
        self.assert_graph(self.G1, self.G2)
        self.assert_object(G_1.has_edge(-1, -1), True)
        self.assert_object(self.G1.has_edge(-1, -1), False)  # type: ignore

        G_1, G_2 = self.run_method("nodes_subgraph", [1, 2, 3, "6"])
        self.assert_graph(G_1, G_2)

        G_1, G_2 = self.run_method("ego_subgraph", 1)
        self.assert_graph(G_1, G_2)

        self.assert_method("__len__")

        self.assert_method("__contains__", 1)
        self.assert_method("__contains__", 10)

        self.assert_method("__getitem__", 1)

        self.assert_method("__iter__")

        print(f"PASSED: Test for {self.class1.__name__} and {self.class2.__name__}")


class TestGraphC(CPPGraphTestBase):
    def setup_method(self, method):
        self.class1 = eg.Graph
        self.class2 = eg.GraphC
        self.G1 = self.class1(name="graph", time=0)  # type: ignore
        self.G2 = self.class2(name="graph", time=0)  # type: ignore


class TestDiGraphC(CPPGraphTestBase):
    def setup_method(self, method):
        self.class1 = eg.DiGraph
        self.class2 = eg.DiGraphC
        self.G1 = self.class1(name="graph", time=0)  # type: ignore
        self.G2 = self.class2(name="graph", time=0)  # type: ignore
