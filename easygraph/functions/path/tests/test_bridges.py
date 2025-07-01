import unittest

import easygraph as eg

from easygraph.utils.exception import EasyGraphNotImplemented


class test_bridges(unittest.TestCase):
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
        self.g4 = eg.Graph()
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

    def result(self, g: eg.Graph):
        res = eg.bridges(g)
        for i in res:
            print(i)

    def test_bridges(self):
        self.result(g=self.g2)
        self.result(g=self.g3)
        self.result(g=self.g4)

    def test_has_bridges(self):
        print(eg.has_bridges(self.g2))

    def test_empty_graph(self):
        g = eg.Graph()
        self.assertFalse(eg.has_bridges(g))
        self.assertEqual(list(eg.bridges(g)), [])

    def test_single_node_graph(self):
        g = eg.Graph()
        g.add_node(1)
        self.assertFalse(eg.has_bridges(g))
        self.assertEqual(list(eg.bridges(g)), [])

    def test_disconnected_graph(self):
        g = eg.Graph()
        g.add_edges_from([(0, 1), (2, 3)])
        self.assertTrue(eg.has_bridges(g))
        self.assertCountEqual(list(eg.bridges(g)), [(0, 1), (2, 3)])

    def test_cycle_graph(self):
        g = eg.DiGraph([(1, 2), (2, 3), (3, 1)])
        self.assertFalse(eg.has_bridges(g))
        self.assertEqual(list(eg.bridges(g)), [])

    def test_path_graph(self):
        g = eg.path_graph(4)
        self.assertTrue(eg.has_bridges(g))
        self.assertCountEqual(list(eg.bridges(g)), [(0, 1), (1, 2), (2, 3)])

    def test_star_graph(self):
        g = eg.Graph()
        g.add_edges_from([(0, i) for i in range(1, 5)])

        expected = [(0, 1), (0, 2), (0, 3), (0, 4)]
        self.assertTrue(eg.has_bridges(g))
        self.assertCountEqual(list(eg.bridges(g)), expected)

    def test_complete_graph(self):
        g = eg.complete_graph(5)
        self.assertFalse(eg.has_bridges(g))
        self.assertEqual(list(eg.bridges(g)), [])

    def test_graph_with_invalid_root(self):
        g = eg.path_graph(3)
        with self.assertRaises(eg.NodeNotFound):
            list(eg.bridges(g, root=10))

    def test_multigraph_exception(self):
        g = eg.MultiGraph()
        g.add_edges_from([(0, 1), (1, 2)])
        with self.assertRaises(EasyGraphNotImplemented):
            list(eg.bridges(g))
        with self.assertRaises(EasyGraphNotImplemented):
            eg.has_bridges(g)

    def test_weighted_graph_should_ignore_weights(self):
        g = eg.Graph()
        g.add_edges_from(
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            edges_attr=[{"weight": 10}, {"weight": 20}, {"weight": 30}, {"weight": 40}],
        )
        self.assertFalse(eg.has_bridges(g))


if __name__ == "__main__":
    unittest.main()
