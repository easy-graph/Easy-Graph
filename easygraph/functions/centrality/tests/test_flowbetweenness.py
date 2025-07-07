import unittest

import easygraph as eg

from easygraph.utils.exception import EasyGraphNotImplemented


class Test_flowbetweenness(unittest.TestCase):
    def setUp(self):
        self.edges = [
            (1, 2),
            (2, 3),
            ("String", "Bool"),
            (2, 1),
            (0, 0),
            (-99, 256),
            ((None, None), (None, None)),
        ]
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

        self.directed_graph = eg.DiGraph()
        self.directed_graph.add_edges_from(
            [
                (0, 1, {"weight": 3}),
                (1, 2, {"weight": 1}),
                (0, 2, {"weight": 1}),
                (2, 3, {"weight": 2}),
                (1, 3, {"weight": 4}),
            ]
        )

        self.graph_with_self_loop = eg.DiGraph()
        self.graph_with_self_loop.add_edges_from([(0, 1), (1, 2), (2, 2), (2, 3)])

        self.disconnected_graph = eg.DiGraph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3)])

        self.undirected_graph = eg.Graph()
        self.undirected_graph.add_edges_from([(0, 1), (1, 2)])

        self.single_node_graph = eg.DiGraph()
        self.single_node_graph.add_node(0)

        self.mixed_type_graph = eg.DiGraph()
        self.mixed_type_graph.add_edges_from([(1, "A"), ("A", (2, 3)), ((2, 3), "B")])

        self.multigraph = eg.MultiDiGraph()
        self.multigraph.add_edges_from([(0, 1), (0, 1)])

    def test_flowbetweenness_centrality(self):
        for i in self.test_graphs:
            print(i.edges)
            print(eg.functions.flowbetweenness_centrality(i))

    def test_flowbetweenness_on_directed(self):
        result = eg.functions.flowbetweenness_centrality(self.directed_graph)
        self.assertIsInstance(result, dict)
        self.assertTrue(
            all(isinstance(v, float) or isinstance(v, int) for v in result.values())
        )

    def test_flowbetweenness_on_self_loop(self):
        result = eg.functions.flowbetweenness_centrality(self.graph_with_self_loop)
        self.assertIsInstance(result, dict)

    def test_flowbetweenness_on_disconnected(self):
        result = eg.functions.flowbetweenness_centrality(self.disconnected_graph)
        self.assertIsInstance(result, dict)

    def test_flowbetweenness_on_single_node(self):
        result = eg.functions.flowbetweenness_centrality(self.single_node_graph)
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {0: 0})

    def test_flowbetweenness_on_mixed_types(self):
        result = eg.functions.flowbetweenness_centrality(self.mixed_type_graph)
        self.assertIsInstance(result, dict)

    def test_flowbetweenness_on_undirected_warns(self):
        result = eg.functions.flowbetweenness_centrality(self.undirected_graph)
        self.assertIsNone(result)

    def test_flowbetweenness_raises_on_multigraph(self):
        with self.assertRaises(EasyGraphNotImplemented):
            eg.functions.flowbetweenness_centrality(self.multigraph)


if __name__ == "__main__":
    unittest.main()
