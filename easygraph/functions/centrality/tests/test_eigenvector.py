import unittest
import numpy as np
import easygraph as eg
from easygraph.functions.centrality import eigenvector_centrality
from easygraph.utils.exception import EasyGraphPointlessConcept, EasyGraphNotImplemented


class Test_eigenvector(unittest.TestCase):
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

        self.test_graphs = []
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

        self.simple_graph = eg.Graph()
        self.simple_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        self.directed_graph = eg.DiGraph()
        self.directed_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        self.weighted_graph = eg.Graph()
        self.weighted_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        for u, v, data in self.weighted_graph.edges:
            data["weight"] = 2.0

        self.disconnected_graph = eg.Graph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3)])

        self.isolated_node_graph = eg.Graph()
        self.isolated_node_graph.add_edges_from([(0, 1), (1, 2)])
        self.isolated_node_graph.add_node(3)  

        self.single_node_graph = eg.Graph()
        self.single_node_graph.add_node(42)

        self.mixed_nodes_graph = eg.Graph()
        self.mixed_nodes_graph.add_edges_from([(1, 2), ("X", "Y"), ((1, 2), (3, 4))])

    def test_eigenvector(self):

        for G in self.test_graphs:
            result = eigenvector_centrality(G)
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), len(G))

            for v in result.values():
                self.assertIsInstance(v, float)

    def test_simple_graph(self):
        result = eigenvector_centrality(self.simple_graph)
        self.assertEqual(len(result), len(self.simple_graph))

        for v in result.values():
            self.assertIsInstance(v, float)

    def test_directed_graph(self):
        result = eigenvector_centrality(self.directed_graph)
        self.assertEqual(len(result), len(self.directed_graph))

    def test_weighted_graph(self):
        result = eigenvector_centrality(self.weighted_graph, weight="weight")
        self.assertEqual(len(result), len(self.weighted_graph))

    def test_disconnected_graph(self):

        result = eigenvector_centrality(self.disconnected_graph)
        self.assertEqual(len(result), len(self.disconnected_graph))

    def test_isolated_nodes(self):

        result = eigenvector_centrality(self.isolated_node_graph)
        self.assertEqual(len(result), len(self.isolated_node_graph))
        self.assertTrue(3 in result)
        self.assertGreaterEqual(result[3], 0)

    def test_single_node_graph(self):
        with self.assertRaises(EasyGraphPointlessConcept):
            eigenvector_centrality(self.single_node_graph)

    def test_mixed_node_types(self):
        result = eigenvector_centrality(self.mixed_nodes_graph)
        self.assertEqual(len(result), len(self.mixed_nodes_graph))

    def test_max_iter_parameter(self):
        result = eigenvector_centrality(self.simple_graph, max_iter=50)
        self.assertEqual(len(result), len(self.simple_graph))

    def test_tol_parameter(self):
        result = eigenvector_centrality(self.simple_graph, tol=1.0e-3)
        self.assertEqual(len(result), len(self.simple_graph))

    def test_nstart_parameter(self):
        nstart = {node: 1.0 for node in self.simple_graph}
        result = eigenvector_centrality(self.simple_graph, nstart=nstart)
        self.assertEqual(len(result), len(self.simple_graph))

    def test_multigraph_raises(self):
        G = eg.MultiGraph()
        G.add_edges_from([(0, 1), (0, 1)])
        with self.assertRaises(EasyGraphNotImplemented):  
            eigenvector_centrality(G)

    def test_empty_graph_raises(self):
        G = eg.Graph()
        with self.assertRaises(EasyGraphPointlessConcept):
            eigenvector_centrality(G)


if __name__ == "__main__":
    unittest.main()