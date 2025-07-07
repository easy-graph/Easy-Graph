import unittest

import easygraph as eg

from easygraph.utils.exception import EasyGraphNotImplemented


class Test_pagerank(unittest.TestCase):
    def setUp(self):
        edges = [
            (1, 2),
            (2, 3),
            ("String", "Bool"),
            (2, 1),
            (0, 0),
            ((None, None), (None, None)),
        ]
        self.g = eg.classes.DiGraph(edges)
        self.directed_graph = eg.DiGraph()
        self.directed_graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

        self.undirected_graph = eg.Graph()
        self.undirected_graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

        self.disconnected_graph = eg.DiGraph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3)])

        self.self_loop_graph = eg.DiGraph()
        self.self_loop_graph.add_edges_from([(0, 0), (0, 1), (1, 2)])

        self.mixed_graph = eg.DiGraph()
        self.mixed_graph.add_edges_from([("A", "B"), ("B", "C"), ("C", (1, 2))])

        self.single_node_graph = eg.DiGraph()
        self.single_node_graph.add_node("solo")

        self.multigraph = eg.MultiDiGraph()
        self.multigraph.add_edges_from([(0, 1), (0, 1)])

    def test_pagerank(self):
        test_graphs = [eg.Graph(), eg.DiGraph()]
        for i in test_graphs:
            print(eg.functions.pagerank(i))

        print(self.g.nodes)
        print(eg.functions.pagerank(self.g))

    """
    def test_google_matrix(self):
        test_graphs = [eg.Graph(), eg.DiGraph(), eg.MultiGraph(), eg.MultiDiGraph()]
        for g in test_graphs:
            print(eg.functions.pagerank.(g))
    """

    def test_directed_graph(self):
        result = eg.functions.pagerank(self.directed_graph)
        self.assertEqual(set(result.keys()), set(self.directed_graph.nodes))

    def test_undirected_graph(self):
        result = eg.functions.pagerank(self.undirected_graph)
        self.assertEqual(set(result.keys()), set(self.undirected_graph.nodes))

    def test_disconnected_graph(self):
        result = eg.functions.pagerank(self.disconnected_graph)
        self.assertEqual(set(result.keys()), set(self.disconnected_graph.nodes))

    def test_self_loop_graph(self):
        result = eg.functions.pagerank(self.self_loop_graph)
        self.assertEqual(set(result.keys()), set(self.self_loop_graph.nodes))

    def test_mixed_node_types(self):
        result = eg.functions.pagerank(self.mixed_graph)
        self.assertEqual(set(result.keys()), set(self.mixed_graph.nodes))

    def test_single_node_graph(self):
        result = eg.functions.pagerank(self.single_node_graph)
        self.assertEqual(result, {"solo": 1.0})

    def test_empty_graph(self):
        empty_graph = eg.DiGraph()
        result = eg.functions.pagerank(empty_graph)
        self.assertEqual(result, {})

    def test_multigraph_raises(self):
        with self.assertRaises(EasyGraphNotImplemented):
            eg.functions.pagerank(self.multigraph)


if __name__ == "__main__":
    unittest.main()
