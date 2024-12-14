import unittest

import easygraph as eg
import easygraph.functions as F
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

    def is_eq(self, edges, nx_cls, eg_cls, k=None):
        G_nx = nx_cls(edges)
        G_eg = eg_cls(edges)

        C_nx = convert_nx_mapping(nx.betweenness_centrality(G_nx, k=k))
        C_eg = F.betweenness_centrality(G_eg, k=k)

        return C_nx == C_eg

    def test_betweenness(self):
        assert self.is_eq([], nx.Graph, eg.Graph)
        assert self.is_eq([], nx.DiGraph, eg.DiGraph)
        assert self.is_eq(self.edges, nx.Graph, eg.Graph)
        assert self.is_eq(self.edges, nx.DiGraph, eg.DiGraph)


if __name__ == "__main__":
    unittest.main()
