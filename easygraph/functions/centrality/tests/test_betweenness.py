import unittest

import easygraph as eg


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
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

        self.undirected = eg.Graph()
        self.undirected.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

        self.directed = eg.DiGraph()
        self.directed.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

        self.disconnected = eg.Graph()
        self.disconnected.add_edges_from([(0, 1), (2, 3)])

        self.single_node = eg.Graph()
        self.single_node.add_node(42)

        self.two_node = eg.Graph()
        self.two_node.add_edge("A", "B")

        self.named_nodes = eg.Graph()
        self.named_nodes.add_edges_from([("X", "Y"), ("Y", "Z")])

    def test_betweenness(self):
        for i in self.test_graphs:
            print(eg.functions.betweenness_centrality(i))

    def test_basic_undirected(self):
        result = eg.functions.betweenness_centrality(self.undirected)
        self.assertEqual(len(result), len(self.undirected.nodes))
        self.assertTrue(all(isinstance(x, float) for x in result))

    def test_basic_directed(self):
        result = eg.functions.betweenness_centrality(self.directed)
        self.assertEqual(len(result), len(self.directed.nodes))

    def test_disconnected(self):
        result = eg.functions.betweenness_centrality(self.disconnected)
        self.assertEqual(len(result), len(self.disconnected.nodes))
        self.assertTrue(all(v == 0.0 for v in result))

    def test_single_node_graph(self):
        result = eg.functions.betweenness_centrality(self.single_node)
        self.assertEqual(result, [0.0])

    def test_two_node_graph(self):
        result = eg.functions.betweenness_centrality(self.two_node)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(v == 0.0 for v in result))

    def test_named_nodes_graph(self):
        result = eg.functions.betweenness_centrality(self.named_nodes)
        self.assertEqual(len(result), 3)

    def test_with_endpoints(self):
        result = eg.functions.betweenness_centrality(self.undirected, endpoints=True)
        self.assertEqual(len(result), len(self.undirected.nodes))

    def test_unormalized(self):
        result = eg.functions.betweenness_centrality(self.undirected, normalized=False)
        self.assertEqual(len(result), len(self.undirected.nodes))

    def test_subset_sources(self):
        result = eg.functions.betweenness_centrality(self.undirected, sources=[1, 2])
        self.assertEqual(len(result), len(self.undirected.nodes))

    def test_parallel_workers(self):
        result = eg.functions.betweenness_centrality(self.undirected, n_workers=2)
        self.assertEqual(len(result), len(self.undirected.nodes))

    def test_multigraph_error(self):
        G = eg.MultiGraph()
        G.add_edges_from([(0, 1), (0, 1)])
        with self.assertRaises(eg.EasyGraphNotImplemented):
            eg.functions.betweenness_centrality(G)

    def test_all_nodes_type_mix(self):
        G = eg.Graph()
        G.add_edges_from([(1, 2), ("A", "B"), ((1, 2), (3, 4))])
        result = eg.functions.betweenness_centrality(G)
        self.assertEqual(len(result), len(G.nodes))


if __name__ == "__main__":
    unittest.main()
