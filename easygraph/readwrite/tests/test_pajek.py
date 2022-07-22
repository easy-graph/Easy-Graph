# # This file is part of the NetworkX distribution.

# NetworkX is distributed with the 3-clause BSD license.


# ::
#    Copyright (C) 2004-2022, NetworkX Developers
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.

#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:

#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#      * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#      * Neither the name of the NetworkX Developers nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Pajek tests
"""
import easygraph as eg


print(eg)
import os
import tempfile

from easygraph.utils import edges_equal
from easygraph.utils import nodes_equal


# from rich import print

test_parse_pajek_edges = [
    (
        "A1",
        "A1",
        0,
        {
            "weight": 1.0,
            "h2": "0",
            "w": "3",
            "c": "Blue",
            "s": "3",
            "a1": "-130",
            "k1": "0.6",
            "a2": "-130",
            "k2": "0.6",
            "ap": "0.5",
            "l": "Bezier loop",
            "lc": "BlueViolet",
            "fos": "20",
            "lr": "58",
            "lp": "0.3",
            "la": "360",
        },
    ),
    (
        "A1",
        "Bb",
        0,
        {
            "weight": 1.0,
            "h2": "0",
            "a1": "40",
            "k1": "2.8",
            "a2": "30",
            "k2": "0.8",
            "ap": "25",
            "l": "Bezier arc",
            "lphi": "90",
            "la": "0",
            "lp": "0.65",
        },
    ),
    (
        "A1",
        "C",
        0,
        {
            "weight": 1.0,
            "p": "Dashed",
            "h2": "0",
            "w": "5",
            "k1": "-1",
            "k2": "-20",
            "ap": "25",
            "l": "Oval arc",
            "c": "Brown",
            "lc": "Black",
        },
    ),
    (
        "Bb",
        "A1",
        0,
        {
            "weight": 1.0,
            "h2": "0",
            "a1": "120",
            "k1": "1.3",
            "a2": "-120",
            "k2": "0.3",
            "ap": "25",
            "l": "Bezier arc",
            "lphi": "270",
            "la": "180",
            "lr": "19",
            "lp": "0.5",
        },
    ),
    (
        "C",
        "D2",
        0,
        {
            "weight": 1.0,
            "p": "Dashed",
            "h2": "0",
            "w": "2",
            "c": "OliveGreen",
            "ap": "25",
            "l": "Straight arc",
            "lc": "PineGreen",
        },
    ),
    (
        "C",
        "C",
        0,
        {
            "weight": -1.0,
            "h1": "6",
            "w": "1",
            "h2": "12",
            "k1": "-2",
            "k2": "-15",
            "ap": "0.5",
            "l": "Circular loop",
            "c": "Red",
            "lc": "OrangeRed",
            "lphi": "270",
            "la": "180",
        },
    ),
    (
        "D2",
        "Bb",
        0,
        {
            "weight": -1.0,
            "h2": "0",
            "w": "1",
            "k1": "-2",
            "k2": "250",
            "ap": "25",
            "l": "Circular arc",
            "c": "Red",
            "lc": "OrangeRed",
        },
    ),
]


class TestPajek:
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
            fh.write(cls.data.encode("UTF-8"))

    @classmethod
    def teardown_class(cls):
        os.unlink(cls.fname)

    def test_parse_pajek_simple(self):
        # Example without node positions or shape
        data = """*Vertices 2\n1 "1"\n2 "2"\n*Edges\n1 2\n2 1"""
        G = eg.parse_pajek(data)
        assert sorted(G.nodes) == ["1", "2"]
        assert edges_equal(G.edges, [("1", "2", 0, {}), ("1", "2", 1, {})])

    def test_parse_pajek(self):
        G = eg.parse_pajek(self.data)
        assert sorted(G.nodes) == ["A1", "Bb", "C", "D2"]
        # print(G.edges)
        assert edges_equal(G.edges, test_parse_pajek_edges)

    def test_parse_pajek_mat(self):
        data = """*Vertices 3\n1 "one"\n2 "two"\n3 "three"\n*Matrix\n1 1 0\n0 1 0\n0 1 0\n"""
        G = eg.parse_pajek(data)
        assert set(G.nodes) == {"one", "two", "three"}
        assert G.nodes["two"] == {"id": "2"}
        assert edges_equal(
            # set(G.edges),
            G.edges,
            [
                ("one", "one", {"weight": 1}),
                ("one", "two", {"weight": 1}),
                ("two", "two", {"weight": 1}),
                ("three", "two", {"weight": 1}),
            ],
        )

    def test_read_pajek(self):
        G = eg.parse_pajek(self.data)
        Gin = eg.read_pajek(self.fname)
        assert sorted(G.nodes) == sorted(Gin.nodes)
        assert edges_equal(G.edges, Gin.edges)
        assert self.G.graph == Gin.graph
        for n in G:
            assert G.nodes[n] == Gin.nodes[n]

    def test_write_pajek(self):
        import io

        G = eg.parse_pajek(self.data)
        fh = io.BytesIO()
        eg.write_pajek(G, fh)
        fh.seek(0)
        H = eg.read_pajek(fh)
        assert nodes_equal(G.nodes, list(H))
        assert edges_equal(G.edges, list(H.edges))
        # Graph name is left out for now, therefore it is not tested.
        # assert_equal(G.graph, H.graph)

    def test_ignored_attribute(self):
        import io

        G = eg.Graph()
        fh = io.BytesIO()
        G.add_node(1, int_attr=1)
        G.add_node(2, empty_attr="  ")
        G.add_edge(1, 2, int_attr=2)
        G.add_edge(2, 3, empty_attr="  ")

        import warnings

        with warnings.catch_warnings(record=True) as w:
            eg.write_pajek(G, fh)
            assert len(w) == 4

    def test_noname(self):
        # Make sure we can parse a line such as:  *network
        # Issue #952
        line = "*network\n"
        other_lines = self.data.split("\n")[1:]
        data = line + "\n".join(other_lines)
        G = eg.parse_pajek(data)

    def test_unicode(self):
        import io

        G = eg.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        G.add_edge(name1, "Radiohead", foo=name2)
        fh = io.BytesIO()
        eg.write_pajek(G, fh)
        fh.seek(0)
        H = eg.read_pajek(fh)
        assert nodes_equal(list(G), list(H))
        # from icecream import ic
        # ic(G.edges)
        # ic(H.edges)
        # ic(G.graph)
        # ic(H.graph)
        # assert edges_equal(list(G.edges), list(H.edges))
        assert G.graph == H.graph
