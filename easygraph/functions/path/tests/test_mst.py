import unittest

import easygraph as eg


class test_mst(unittest.TestCase):
    def setUp(self):
        self.g1 = eg.get_graph_karateclub()

        # source graph: https://zh.wikipedia.org/zh-cn/%E6%88%B4%E5%85%8B%E6%96%AF%E7%89%B9%E6%8B%89%E7%AE%97%E6%B3%95#/media/File:Dijkstra_Animation.gif
        edges = [(1, 2), (1, 3), (1, 6), (2, 3), (2, 4), (3, 4), (3, 6), (4, 5), (5, 6)]
        self.g2 = eg.Graph(edges)
        self.g2.add_edges(
            edges,
            edges_attr=[
                {"weight": 7},
                {"weight": 9},
                {"weight": 14},
                {"weight": 10},
                {"weight": 15},
                {"weight": 11},
                {"weight": 2},
                {"weight": 6},
                {"weight": 9},
            ],
        )

        # source graph: https://static.javatpoint.com/tutorial/daa/images/dijkstra-algorithm.png
        self.g3 = eg.Graph()
        edges = [
            (0, 1),
            (0, 4),
            (1, 4),
            (1, 2),
            (4, 5),
            (4, 8),
            (2, 3),
            (2, 6),
            (2, 8),
            (5, 6),
            (5, 8),
            (3, 6),
            (3, 7),
            (6, 7),
        ]

        self.g3.add_edges(
            edges,
            edges_attr=[
                {"weight": 4},
                {"weight": 1},
                {"weight": 11},
                {"weight": 8},
                {"weight": 1},
                {"weight": 7},
                {"weight": 7},
                {"weight": 4},
                {"weight": 2},
                {"weight": 2},
                {"weight": 6},
                {"weight": 14},
                {"weight": 9},
                {"weight": 10},
            ],
        )
        self.g4 = eg.DiGraph()
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        self.g4.add_edges(
            edges,
            edges_attr=[
                {"weight": -1},
                {"weight": -2},
                {"weight": -3},
                {"weight": -4},
                {"weight": -5},
                {"weight": -6},
            ],
        )
        self.nan_graph = eg.Graph()
        self.nan_graph.add_edges(
            [(0, 1), (1, 2)], edges_attr=[{"weight": float("nan")}, {"weight": 1}]
        )

        self.no_weight_graph = eg.Graph()
        self.no_weight_graph.add_edges([(0, 1), (1, 2)])

        self.equal_weight_graph = eg.Graph()
        self.equal_weight_graph.add_edges(
            [(0, 1), (1, 2), (2, 0)],
            edges_attr=[{"weight": 1}, {"weight": 1}, {"weight": 1}],
        )

        self.negative_weight_graph = eg.Graph()
        self.negative_weight_graph.add_edges(
            [(0, 1), (1, 2), (2, 3)],
            edges_attr=[{"weight": -1}, {"weight": -2}, {"weight": -3}],
        )

        self.disconnected_graph = eg.Graph()
        self.disconnected_graph.add_edges(
            [(0, 1), (2, 3)], edges_attr=[{"weight": 1}, {"weight": 2}]
        )

        self.G = eg.Graph()
        self.G.add_edges(
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            edges_attr=[{"weight": 1}, {"weight": 2}, {"weight": 3}, {"weight": 4}],
        )

    def helper(self, g: eg.Graph, func):
        result = func(g)
        if isinstance(result, eg.Graph):
            print("nodes: " + str(result.nodes))
            print("edges: " + str(result.edges))
        else:
            for i in result:
                print(i)

    def test_minimum_spanning_edges(self):
        print("test_minimum_spanning_edges")
        self.helper(self.g2, eg.minimum_spanning_edges)
        self.helper(self.g2, eg.minimum_spanning_edges)
        self.helper(self.g4, eg.minimum_spanning_edges)

    def test_maximum_spanning_edges(self):
        print("test_maximum_spanning_edges")
        self.helper(self.g2, eg.maximum_spanning_edges)
        self.helper(self.g2, eg.maximum_spanning_edges)
        self.helper(self.g4, eg.maximum_spanning_edges)

    def test_minimum_spanning_tree(self):
        print("test_minimum_spanning_tree")
        self.helper(self.g2, eg.minimum_spanning_tree)
        self.helper(self.g2, eg.minimum_spanning_tree)
        self.helper(self.g4, eg.minimum_spanning_tree)

    def test_maximum_spanning_tree(self):
        print("test_maximum_spanning_tree")
        self.helper(self.g2, eg.maximum_spanning_tree)
        self.helper(self.g2, eg.maximum_spanning_tree)
        self.helper(self.g4, eg.maximum_spanning_tree)

    def test_nan_handling(self):
        with self.assertRaises(ValueError):
            list(eg.minimum_spanning_edges(self.nan_graph))
        edges = list(eg.minimum_spanning_edges(self.nan_graph, ignore_nan=True))
        self.assertEqual(len(edges), 1)

    def test_missing_weight_defaults_to_one(self):
        edges = list(eg.minimum_spanning_edges(self.no_weight_graph))
        self.assertEqual(len(edges), 2)

    def test_negative_weights(self):
        edges = list(eg.minimum_spanning_edges(self.negative_weight_graph))
        weights = [attr["weight"] for _, _, attr in edges]
        self.assertIn(-3, weights)
        self.assertEqual(len(edges), 3)

    def test_disconnected_graph(self):
        edges = list(eg.minimum_spanning_edges(self.disconnected_graph))
        self.assertEqual(len(edges), 2)

    def test_maximum_vs_minimum_edges(self):
        min_edges = list(eg.minimum_spanning_edges(self.G))
        max_edges = list(eg.maximum_spanning_edges(self.G))
        min_set = {(min(u, v), max(u, v)) for u, v, _ in min_edges}
        max_set = {(min(u, v), max(u, v)) for u, v, _ in max_edges}
        self.assertNotEqual(min_set, max_set)

    def test_invalid_algorithm_name(self):
        with self.assertRaises(ValueError):
            list(eg.minimum_spanning_edges(self.G, algorithm="invalid_algo"))


if __name__ == "__main__":
    unittest.main()
