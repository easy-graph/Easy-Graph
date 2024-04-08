"""
UCINET tests
"""

import io

import easygraph as eg


# from nose import SkipTest
# from nose.tools import *


def filterEdges(edges):
    return [e[:3] for e in edges]


class TestUcinet:
    @classmethod
    def setup_class(self):
        self.G = eg.MultiDiGraph()
        self.G.add_nodes_from(["a", "b", "c", "d", "e"])
        self.G.add_edges_from(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("a", "e"),
                ("b", "a"),
                ("b", "c"),
                ("b", "d"),
                ("c", "a"),
                ("c", "b"),
                ("d", "a"),
                ("d", "b"),
                ("e", "a"),
            ]
        )
        try:
            pass
        except ImportError:
            print("NumPy not available.")
            # raise SkipTest("NumPy not available.")

    def test_generate_ucinet(self):
        Gout = eg.generate_ucinet(self.G)
        s = ""
        for line in Gout:
            s += line + "\n"
        G_generated = eg.parse_ucinet(s)

        data = """\
dl n=5 format=fullmatrix
labels:
a,b,c,d,e
data:
0 1 1 1 1
1 0 1 1 0
1 1 0 0 0
1 1 0 0 0
1 0 0 0 0"""
        G = eg.parse_ucinet(data)
        assert sorted(G.nodes) == sorted(G_generated.nodes)
        assert sorted(G.edges) == sorted(G_generated.edges)

    def test_parse_ucinet(self):
        data = """
DL N = 5
Data:
0 1 1 1 1
1 0 1 0 0
1 1 0 0 1
1 0 0 0 0
1 0 1 0 0
        """
        graph = eg.MultiDiGraph()
        graph.add_nodes_from([0, 1, 2, 3, 4])
        graph.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 4),
                (3, 0),
                (4, 0),
                (4, 2),
            ]
        )
        G = eg.parse_ucinet(data)
        assert sorted(G.nodes) == sorted(graph.nodes)
        assert sorted(filterEdges(G.edges)) == sorted(filterEdges(graph.edges))
        # print [n for n in G.nodes(data=True)]
        # print [e for e in G.edges]

    def test_parse_ucinet_labels(self):
        """
        Test parsing of labels : single line (data1), multiple lines (data2), embedded (data3)
        Labels must be separated by spaces, carriage returns, equal signs or commas.
        Labels with embedded spaces are not advisable, but can be entered by
        surrounding the label in quotes (e.g., "Humpty Dumpty").
        """
        data1 = """
dl n=5
format = fullmatrix
labels:
barry,david,lin,pat,russ
data:
0 1 1 1 0
1 0 0 0 1
1 0 0 1 0
1 0 1 0 1
0 1 0 1 0
                """
        data2 = """
dl n=5
format = fullmatrix
labels:
barry,david
lin,pat
russ
data:
0 1 1 1 0
1 0 0 0 1
1 0 0 1 0
1 0 1 0 1
0 1 0 1 0
        """
        data3 = """\
dl n=5
format = fullmatrix
labels embedded
data:
barry david lin pat russ
Barry 0 1 1 1 0
david 1 0 0 0 1
Lin 1 0 0 1 0
Pat 1 0 1 0 1
Russ 0 1 0 1 0
        """
        G = eg.MultiDiGraph()
        G.add_nodes_from(["russ", "barry", "lin", "pat", "david"])
        G.add_edges_from(
            [
                ("russ", "pat"),
                ("russ", "david"),
                ("barry", "lin"),
                ("barry", "pat"),
                ("barry", "david"),
                ("lin", "barry"),
                ("lin", "pat"),
                ("pat", "barry"),
                ("pat", "lin"),
                ("pat", "russ"),
                ("david", "barry"),
                ("david", "russ"),
            ]
        )
        G1 = eg.parse_ucinet(data1)
        G2 = eg.parse_ucinet(data2)
        G3 = eg.parse_ucinet(data3)
        assert sorted(G1.nodes) == sorted(G.nodes)
        assert sorted(G2.nodes) == sorted(G.nodes)
        assert sorted(G3.nodes) == sorted(G.nodes)
        assert sorted(e[:3] for e in G1.edges) == sorted(e[:3] for e in G.edges)
        assert sorted(e[:3] for e in G2.edges) == sorted(e[:3] for e in G.edges)
        assert sorted(e[:3] for e in G3.edges) == sorted(e[:3] for e in G.edges)
        # print [n for n in G.nodes]
        # print [e for e in G.edges]

    def test_parse_ucinet_nodelist1(self):
        data1 = """
DL n=4
format = nodelist1
data:
  1  3 2 1
  4  1 4
  2  2 4 1
        """
        data2 = """
DL n=4
format = nodelist1b
data:
  3  1 2 3
  3  1 2 4
  0
  2  1 4
        """
        G = eg.MultiDiGraph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from(
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3), (3, 0), (3, 3)]
        )
        G1 = eg.parse_ucinet(data1)
        G2 = eg.parse_ucinet(data2)
        assert sorted(G1.nodes) == sorted(G.nodes)
        assert sorted(G2.nodes) == sorted(G.nodes)
        assert sorted(filterEdges(G1.edges)) == sorted(filterEdges(G.edges))
        assert sorted(filterEdges(G2.edges)) == sorted(filterEdges(G.edges))

    def test_parse_ucinet_nodelist1_labels(self):
        data1 = """
DL n=5
format = nodelist1
labels:
george, sally, jim, billy, jane
data:
1 2 3
2 3
4 1
5 3
        """
        data2 = """
DL n=5
format = nodelist1
labels embedded:
data:
george sally jim
sally jim
billy george
jane jim
        """
        G = eg.MultiDiGraph()
        G.add_nodes_from(["george", "sally", "jim", "billy", "jane"])
        G.add_edges_from(
            [
                ("billy", "george"),
                ("jane", "jim"),
                ("sally", "jim"),
                ("george", "jim"),
                ("george", "sally"),
            ]
        )
        G1 = eg.parse_ucinet(data1)
        G2 = eg.parse_ucinet(data2)
        assert sorted(G1.nodes) == sorted(G.nodes)
        assert sorted(G2.nodes) == sorted(G.nodes)
        assert sorted(G1.edges) == sorted(G.edges)
        assert sorted(G2.edges) == sorted(G.edges)

    def test_read_ucinet(self):
        fh = io.BytesIO()
        data = """
DL N = 5
Data:
0 1 1 1 1
1 0 1 0 0
1 1 0 0 1
1 0 0 0 0
1 0 1 0 0
        """
        Gin = eg.parse_ucinet(data)
        fh.write(data.encode("UTF-8"))
        fh.seek(0)
        Gout = eg.read_ucinet(fh)
        assert sorted(Gout.nodes) == sorted(Gin.nodes)
        assert sorted(e[:3] for e in Gout.edges) == sorted(e[:3] for e in Gin.edges)

    def test_write_ucinet(self):
        fh = io.BytesIO()
        data = """\
dl n=5 format=fullmatrix
data:
0 1 1 1 1
1 0 1 0 0
1 1 0 0 1
1 0 0 0 0
1 0 1 0 0
"""
        graph = eg.MultiDiGraph()
        graph.add_nodes_from([0, 1, 2, 3, 4])
        graph.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 4),
                (3, 0),
                (4, 0),
                (4, 2),
            ]
        )

        eg.write_ucinet(graph, fh)
        fh.seek(0)
        G = eg.parse_ucinet(fh.readlines())
        assert sorted(G.nodes) == sorted(graph.nodes)
        assert sorted(e[:3] for e in G.edges) == sorted(e[:3] for e in graph.edges)

    def test_parse_ucinet_edgelist1(self):
        data1 = """
DL n=5
format = edgelist1
labels:
george, sally, jim, billy, jane
data:
1 2
1 3
2 3
3 1
5 4
        """
        data2 = """
DL n=5
format = edgelist1
labels embedded:
data:
george sally
george jim
sally jim
jim george
jane billy
        """
        G = eg.MultiDiGraph()
        G.add_nodes_from(["george", "sally", "jim", "billy", "jane"])
        G.add_edges_from(
            [
                ("jim", "george"),
                ("jane", "billy"),
                ("sally", "jim"),
                ("george", "jim"),
                ("george", "sally"),
            ]
        )

        G1 = eg.parse_ucinet(data1)
        G2 = eg.parse_ucinet(data2)
        assert sorted(G1.nodes) == sorted(G.nodes)
        assert sorted(G2.nodes) == sorted(G.nodes)
        assert sorted(G1.edges) == sorted(G.edges)
        assert sorted(G2.edges) == sorted(G.edges)
