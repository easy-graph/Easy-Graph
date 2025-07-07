import unittest

import easygraph as eg

from easygraph.utils.exception import EasyGraphNotImplemented


class Test_egobetweenness(unittest.TestCase):
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
        print(self.test_graphs[-1].edges)

        self.graph = eg.Graph()
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        self.directed_graph = eg.DiGraph()
        self.directed_graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

        self.mixed_nodes_graph = eg.Graph()
        self.mixed_nodes_graph.add_edges_from([(1, "A"), ("A", (2, 3)), ((2, 3), "B")])

        self.single_node_graph = eg.Graph()
        self.single_node_graph.add_node(42)

        self.disconnected_graph = eg.Graph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3)])  # two components

        self.multigraph = eg.MultiGraph()
        self.multigraph.add_edges_from([(0, 1), (0, 1)])  # parallel edges

    def test_egobetweenness(self):
        print(eg.functions.ego_betweenness(self.test_graphs[-1], 4))

    def test_small_undirected_graph(self):
        result = eg.functions.ego_betweenness(self.graph, 1)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_directed_graph(self):
        result = eg.functions.ego_betweenness(self.directed_graph, 0)
        self.assertIsInstance(result, int)

    def test_mixed_node_types(self):
        result = eg.functions.ego_betweenness(self.mixed_nodes_graph, "A")
        self.assertIsInstance(result, float)

    def test_single_node_graph(self):
        result = eg.functions.ego_betweenness(self.single_node_graph, 42)
        self.assertEqual(result, 0.0)

    def test_disconnected_graph_component(self):
        result_0 = eg.functions.ego_betweenness(self.disconnected_graph, 0)
        result_2 = eg.functions.ego_betweenness(self.disconnected_graph, 2)
        self.assertIsInstance(result_0, float)
        self.assertIsInstance(result_2, float)

    def test_raises_on_multigraph(self):
        with self.assertRaises(EasyGraphNotImplemented):
            eg.functions.ego_betweenness(self.multigraph, 0)


if __name__ == "__main__":
    unittest.main()
