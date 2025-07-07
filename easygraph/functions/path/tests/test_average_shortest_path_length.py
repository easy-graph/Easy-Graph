import unittest

import easygraph as eg

from easygraph import average_shortest_path_length
from easygraph.utils.exception import EasyGraphError
from easygraph.utils.exception import EasyGraphPointlessConcept


class TestAverageShortestPathLength(unittest.TestCase):
    def test_unweighted_path_graph(self):
        G = eg.path_graph(5)
        result = average_shortest_path_length(G)
        self.assertEqual(result, 2.0)

    def test_weighted_graph(self):
        G = eg.Graph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=2)
        G.add_edge(2, 3, weight=3)
        result = average_shortest_path_length(G, weight="weight", method="dijkstra")
        self.assertAlmostEqual(result, 3.333, places=3)

    def test_trivial_graph(self):
        G = eg.Graph()
        G.add_node(1)
        self.assertEqual(average_shortest_path_length(G), 0)

    def test_disconnected_graph_undirected(self):
        G = eg.Graph([(1, 2), (3, 4)])
        with self.assertRaises(EasyGraphError):
            average_shortest_path_length(G)

    def test_disconnected_graph_directed(self):
        G = eg.DiGraph([(0, 1), (2, 3)])
        with self.assertRaises(EasyGraphError):
            average_shortest_path_length(G)

    def test_null_graph(self):
        G = eg.Graph()
        with self.assertRaises(EasyGraphPointlessConcept):
            average_shortest_path_length(G)

    def test_directed_strongly_connected(self):
        G = eg.DiGraph([(0, 1), (1, 2), (2, 0)])
        result = average_shortest_path_length(G)
        self.assertEqual(result, 1.5)

    def test_unsupported_method(self):
        G = eg.path_graph(5)
        with self.assertRaises(ValueError):
            average_shortest_path_length(G, method="unsupported_method")


if __name__ == "__main__":
    unittest.main()
