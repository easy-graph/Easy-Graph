#!/usr/bin/env python3
"""
pickle read / write tests
"""

import os
import pickle
import tempfile

import easygraph as eg

from easygraph.utils import edges_equal


class TestPickle:
    @classmethod
    def setup_class(cls):
        cls.data = """*network Tralala\n*vertices 4\n   1 "A1"         0.0938 0.0896   ellipse x_fact 1 y_fact 1\n   2 "Bb"         0.8188 0.2458   ellipse x_fact 1 y_fact 1\n   3 "C"          0.3688 0.7792   ellipse x_fact 1\n   4 "D2"         0.9583 0.8563   ellipse x_fact 1\n*arcs\n1 1 1  h2 0 w 3 c Blue s 3 a1 -130 k1 0.6 a2 -130 k2 0.6 ap 0.5 l "Bezier loop" lc BlueViolet fos 20 lr 58 lp 0.3 la 360\n2 1 1  h2 0 a1 120 k1 1.3 a2 -120 k2 0.3 ap 25 l "Bezier arc" lphi 270 la 180 lr 19 lp 0.5\n1 2 1  h2 0 a1 40 k1 2.8 a2 30 k2 0.8 ap 25 l "Bezier arc" lphi 90 la 0 lp 0.65\n4 2 -1  h2 0 w 1 k1 -2 k2 250 ap 25 l "Circular arc" c Red lc OrangeRed\n3 4 1  p Dashed h2 0 w 2 c OliveGreen ap 25 l "Straight arc" lc PineGreen\n1 3 1  p Dashed h2 0 w 5 k1 -1 k2 -20 ap 25 l "Oval arc" c Brown lc Black\n3 3 -1  h1 6 w 1 h2 12 k1 -2 k2 -15 ap 0.5 l "Circular loop" c Red lc OrangeRed lphi 270 la 180"""
        cls.G = eg.MultiDiGraph()
        cls.G.add_nodes_from(["A1", "Bb", "C", "D2"])
        cls.G.add_edges_from(
            [
                ("A1", "A1"),
                ("A1", "Bb"),
                ("A1", "C"),
                ("Bb", "A1"),
                ("C", "C"),
                ("C", "D2"),
                ("D2", "Bb"),
            ]
        )

        cls.G.graph["name"] = "Tralala"
        (fd, cls.fname) = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as fh:
            fh.write(pickle.dumps(cls.G))

    @classmethod
    def teardown_class(cls):
        os.unlink(cls.fname)

    def test_read_pickle(self):
        G = eg.read_pickle(self.fname)
        assert G.nodes == self.G.nodes
        assert G.edges == self.G.edges

    def test_write_pickle(self):
        G = eg.parse_pajek(self.data)
        eg.write_pickle(self.fname, G)
        Gin = eg.read_pickle(self.fname)
        assert sorted(G.nodes) == sorted(Gin.nodes)
        assert edges_equal(G.edges, Gin.edges)
        assert self.G.graph == Gin.graph
