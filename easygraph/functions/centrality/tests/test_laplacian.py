import unittest

import easygraph as eg

from easygraph.utils.exception import EasyGraphNotImplemented


class Test_laplacian(unittest.TestCase):
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
        self.weighted_graph = eg.Graph()
        self.weighted_graph.add_edges_from(
            [
                (0, 1, {"weight": 2}),
                (1, 2, {"weight": 3}),
                (2, 3, {"weight": 4}),
                (3, 0, {"weight": 1}),
            ]
        )

        self.unweighted_graph = eg.Graph()
        self.unweighted_graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
            ]
        )

        self.directed_graph = eg.DiGraph()
        self.directed_graph.add_edges_from(
            [
                (0, 1, {"weight": 2}),
                (1, 2, {"weight": 1}),
                (2, 0, {"weight": 3}),
            ]
        )

        self.self_loop_graph = eg.Graph()
        self.self_loop_graph.add_edges_from(
            [
                (0, 0, {"weight": 2}),
                (0, 1, {"weight": 1}),
            ]
        )

        self.mixed_type_graph = eg.Graph()
        self.mixed_type_graph.add_edges_from(
            [
                ("A", "B"),
                ("B", (1, 2)),
            ]
        )

        self.single_node_graph = eg.Graph()
        self.single_node_graph.add_node(42)

        self.multigraph = eg.MultiGraph()
        self.multigraph.add_edges_from([(0, 1), (0, 1)])

    def test_laplacian(self):
        for i in self.test_graphs:
            print(i.edges)
            print(eg.functions.laplacian(i))

    def test_weighted_graph(self):
        result = eg.functions.laplacian(self.weighted_graph)
        self.assertEqual(set(result.keys()), set(self.weighted_graph.nodes))

    def test_unweighted_graph(self):
        result = eg.functions.laplacian(self.unweighted_graph)
        self.assertEqual(set(result.keys()), set(self.unweighted_graph.nodes))

    def test_directed_graph(self):
        result = eg.functions.laplacian(self.directed_graph)
        self.assertEqual(set(result.keys()), set(self.directed_graph.nodes))

    def test_self_loop_graph(self):
        result = eg.functions.laplacian(self.self_loop_graph)
        self.assertEqual(set(result.keys()), set(self.self_loop_graph.nodes))

    def test_mixed_node_types(self):
        result = eg.functions.laplacian(self.mixed_type_graph)
        self.assertEqual(set(result.keys()), set(self.mixed_type_graph.nodes))

    def test_single_node_graph(self):
        result = eg.functions.laplacian(self.single_node_graph)
        self.assertEqual(result, {})

    def test_multigraph_raises(self):
        with self.assertRaises(EasyGraphNotImplemented):
            eg.functions.laplacian(self.multigraph)


if __name__ == "__main__":
    unittest.main()
