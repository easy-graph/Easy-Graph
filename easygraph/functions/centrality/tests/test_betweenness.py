import unittest

import easygraph as eg
import networkx as nx
import numpy as np


def convert_nx_mapping(mapping):
    return list(mapping.values())


class Test_betweenness(unittest.TestCase):
    def setUp(self):
        self.edges = [
            (1, 4),
            (2, 4),
            ("String", "Bool"),
            (4, 1),
            (0, 4),
            (4, 256),
            ((None, None), (None, None)),
        ]

    def is_eq(self, edges, nx_cls, eg_cls, **kwargs):
        G_nx = nx_cls(edges)
        G_eg = eg_cls(edges)

        C_nx = convert_nx_mapping(nx.betweenness_centrality(G_nx, **kwargs))
        C_eg = eg.functions.betweenness_centrality(G_eg, **kwargs)

        np.testing.assert_almost_equal(C_nx, C_eg)

    def is_cpp_eq(self, edges, nx_cls, eg_cls, **kwargs):
        G_nx = nx_cls(edges)
        G_eg = eg_cls(edges)

        C_nx = convert_nx_mapping(nx.betweenness_centrality(G_nx, **kwargs))
        C_eg = eg.functions.betweenness_centrality(G_eg.cpp(), **kwargs)

        np.testing.assert_almost_equal(C_nx, C_eg)

    def test_betweenness(self):
        self.is_eq([], nx.Graph, eg.Graph)
        self.is_eq([], nx.DiGraph, eg.DiGraph)
        self.is_eq(self.edges, nx.Graph, eg.Graph)
        self.is_eq(self.edges, nx.DiGraph, eg.DiGraph)
        self.is_eq(self.edges, nx.Graph, eg.Graph, k=3, seed=42)
        self.is_eq(self.edges, nx.DiGraph, eg.DiGraph, k=3, seed=42)
        self.is_eq(self.edges, nx.Graph, eg.Graph, k=5, seed=42)
        self.is_eq(self.edges, nx.DiGraph, eg.DiGraph, k=5, seed=42)

    def test_cpp_betweenness(self):
        self.is_cpp_eq([], nx.Graph, eg.Graph)
        self.is_cpp_eq([], nx.DiGraph, eg.DiGraph)
        self.is_cpp_eq(self.edges, nx.Graph, eg.Graph)
        self.is_cpp_eq(self.edges, nx.DiGraph, eg.DiGraph)
        self.is_cpp_eq(self.edges, nx.Graph, eg.Graph, k=3, seed=42)
        self.is_cpp_eq(self.edges, nx.DiGraph, eg.DiGraph, k=3, seed=42)
        self.is_cpp_eq(self.edges, nx.Graph, eg.Graph, k=5, seed=42)
        self.is_cpp_eq(self.edges, nx.DiGraph, eg.DiGraph, k=5, seed=42)


if __name__ == "__main__":
    unittest.main()
