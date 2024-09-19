import unittest

import easygraph as eg


class test_path(unittest.TestCase):
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

    def test_Dijkstra(self):
        print("Dijkstra tested:")
        print(eg.Dijkstra(self.g2, node=1))
        print(eg.Dijkstra(self.g3, node=0))
        print()

    def test_Floyd(self):
        print("Floyd tested:")
        print(eg.Floyd(self.g2))
        print(eg.Floyd(self.g3))
        print(eg.Floyd(self.g4))
        # probably need a negative circle detection for input graphs
        print()

    def test_Prim(self):
        print("Prim tested:")
        print(eg.Prim(self.g2))
        print(eg.Prim(self.g3))
        print(eg.Prim(self.g4))
        print()

    def test_Kruskal(self):
        print("Krusal tested:")
        print(eg.Kruskal(self.g2))
        print(eg.Kruskal(self.g3))
        print(eg.Kruskal(self.g4))
        print()

    def test_Spfa(self):
        try:
            print(eg.Spfa(self.g2, 1))
        except eg.EasyGraphError as e:
            print(e)

    def test_single_source_bfs(self):
        print("single_source_bfs tested:")
        print(eg.single_source_bfs(self.g2, 1, target=5))
        print(eg.single_source_bfs(self.g3, 0, target=6))
        print(eg.single_source_bfs(self.g4, 0, target=3))
        print()

    def test_single_source_dijkstra(self):
        # identical to Dijkstra method
        pass

    def test_multi_source_dijkstra(self):
        print("multi_source_dijkstra tested")
        print(eg.multi_source_dijkstra(self.g2, sources=list(self.g2.nodes.keys())))
        print(eg.multi_source_dijkstra(self.g3, sources=list(self.g2.nodes.keys())))
        try:
            print(eg.multi_source_dijkstra(self.g4, sources=list(self.g2.nodes.keys())))
        except ValueError as e:
            print(e)
        print()


if __name__ == "__main__":
    unittest.main()
