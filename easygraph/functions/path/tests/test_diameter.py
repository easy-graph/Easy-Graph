import unittest

import easygraph as eg


class test_diameter(unittest.TestCase):
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
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (1, 0)]
        self.g4.add_edges(
            edges,
            edges_attr=[
                {"weight": 1},
                {"weight": 2},
                {"weight": 3},
                {"weight": 4},
                {"weight": 5},
                {"weight": 6},
                {"weight": 11},
            ],
        )

    def test_diameter(self):
        print(eg.diameter(self.g2))
        print(eg.diameter(self.g3))
        print(eg.diameter(self.g4))

    def test_eccentricity(self):
        print(eg.eccentricity(self.g2, list(self.g2.nodes.keys())[0:-1]))
        print(eg.eccentricity(self.g3))
        print(eg.eccentricity(self.g4))

    def test_single_node_graph(self):
        G = eg.Graph()
        G.add_node(1)
        self.assertEqual(eg.eccentricity(G), {1: 0})
        self.assertEqual(eg.diameter(G), 0)

    def test_two_node_graph(self):
        G = eg.Graph([(1, 2)])
        self.assertEqual(eg.eccentricity(G), {1: 1, 2: 1})
        self.assertEqual(eg.diameter(G), 1)

    def test_disconnected_graph(self):
        G = eg.Graph()
        G.add_nodes_from([1, 2, 3])
        G.add_edge(1, 2)
        with self.assertRaises(eg.EasyGraphError):
            eg.eccentricity(G)

    def test_directed_not_strongly_connected(self):
        G = eg.DiGraph()
        G.add_edges_from([(1, 2), (2, 3)])  # Not strongly connected
        with self.assertRaises(eg.EasyGraphError):
            eg.eccentricity(G)

    def test_eccentricity_with_sp(self):
        G = eg.Graph([(1, 2), (2, 3)])
        sp = {
            1: {1: 0, 2: 1, 3: 2},
            2: {2: 0, 1: 1, 3: 1},
            3: {3: 0, 2: 1, 1: 2},
        }
        self.assertEqual(eg.eccentricity(G, sp=sp), {1: 2, 2: 1, 3: 2})
        self.assertEqual(eg.diameter(G, e=eg.eccentricity(G, sp=sp)), 2)

    def test_eccentricity_single_node_query(self):
        G = eg.Graph([(1, 2), (2, 3)])
        self.assertEqual(eg.eccentricity(G, v=1), 2)
        self.assertEqual(eg.eccentricity(G, v=2), 1)

    def test_eccentricity_subset_of_nodes(self):
        G = eg.Graph([(1, 2), (2, 3)])
        result = eg.eccentricity(G, v=[1, 3])
        self.assertEqual(result[1], 2)
        self.assertEqual(result[3], 2)

    def test_diameter_matches_max_eccentricity(self):
        G = eg.Graph([(1, 2), (2, 3)])
        ecc = eg.eccentricity(G)
        self.assertEqual(eg.diameter(G, e=ecc), max(ecc.values()))


if __name__ == "__main__":
    unittest.main()
