import os
import tempfile

import pytest


pygraphviz = pytest.importorskip("pygraphviz")

import easygraph as eg

from easygraph.utils import edges_equal
from easygraph.utils import nodes_equal


class TestAGraph:
    def build_graph(self, G):
        edges = [("A", "B"), ("A", "C"), ("A", "C"), ("B", "C"), ("A", "D")]
        G.add_edges_from(edges)
        G.add_node("E")
        G.graph["metal"] = "bronze"
        return G

    def assert_equal(self, G1, G2):
        assert nodes_equal(G1.nodes, G2.nodes)
        assert edges_equal(G1.edges, G2.edges)
        assert G1.graph["metal"] == G2.graph["metal"]

    def agraph_checks(self, G):
        G = self.build_graph(G)
        A = eg.to_agraph(G)
        H = eg.from_agraph(A)
        self.assert_equal(G, H)

        fd, fname = tempfile.mkstemp()
        eg.write_dot(H, fname)
        Hin = eg.read_dot(fname)
        self.assert_equal(H, Hin)
        os.close(fd)
        os.unlink(fname)

        (fd, fname) = tempfile.mkstemp()
        with open(fname, "w") as fh:
            eg.write_dot(H, fh)

        with open(fname) as fh:
            Hin = eg.read_dot(fh)
        os.close(fd)
        os.unlink(fname)
        self.assert_equal(H, Hin)

    def test_from_agraph_name(self):
        G = eg.Graph(name="test")
        A = eg.to_agraph(G)
        H = eg.from_agraph(A)
        assert G.name == "test"

    def test_undirected(self):
        self.agraph_checks(eg.Graph())
