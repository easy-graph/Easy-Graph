import unittest

import easygraph as eg

from easygraph.classes.multigraph import MultiGraph
from easygraph.functions.centrality import closeness_centrality


class Test_closeness(unittest.TestCase):
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
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

        self.simple_graph = eg.Graph()
        self.simple_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        self.directed_graph = eg.DiGraph()
        self.directed_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        self.weighted_graph = eg.Graph()
        self.weighted_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        for u, v, data in self.weighted_graph.edges:
            data["weight"] = 2

        self.disconnected_graph = eg.Graph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3)])

        self.single_node_graph = eg.Graph()
        self.single_node_graph.add_node(42)

        self.mixed_nodes_graph = eg.Graph()
        self.mixed_nodes_graph.add_edges_from([(1, 2), ("X", "Y"), ((1, 2), (3, 4))])

    def test_closeness(self):
        for i in self.test_graphs:
            result = closeness_centrality(i)
            self.assertEqual(len(result), len(i))

    def test_simple_graph(self):
        result = closeness_centrality(self.simple_graph)
        self.assertEqual(len(result), len(self.simple_graph))
        self.assertTrue(all(isinstance(x, float) for x in result))

    def test_directed_graph(self):
        result = closeness_centrality(self.directed_graph)
        self.assertEqual(len(result), len(self.directed_graph))

    def test_weighted_graph(self):
        result = closeness_centrality(self.weighted_graph, weight="weight")
        self.assertEqual(len(result), len(self.weighted_graph))

    def test_disconnected_graph(self):
        result = closeness_centrality(self.disconnected_graph)
        self.assertEqual(len(result), len(self.disconnected_graph))
        self.assertTrue(all(v <= 1.0 for v in result))

    def test_single_node_graph(self):
        result = closeness_centrality(self.single_node_graph)
        self.assertEqual(result, [0.0])

    def test_mixed_node_types(self):
        result = closeness_centrality(self.mixed_nodes_graph)
        self.assertEqual(len(result), len(self.mixed_nodes_graph))

    def test_parallel_workers(self):
        result = closeness_centrality(self.simple_graph, n_workers=2)
        self.assertEqual(len(result), len(self.simple_graph))

    def test_multigraph_raises(self):
        G = MultiGraph()
        G.add_edges_from([(0, 1), (0, 1)])
        with self.assertRaises(eg.EasyGraphNotImplemented):
            closeness_centrality(G)


if __name__ == "__main__":
    unittest.main()
