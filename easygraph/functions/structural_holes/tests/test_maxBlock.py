import random
import unittest

import easygraph as eg


class TestMaxBlockMethods(unittest.TestCase):
    def setUp(self):
        self.G = eg.DiGraph()
        self.G.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),  # Strongly connected
                (2, 3),
                (3, 4),
                (4, 2),  # Another cycle
            ]
        )
        for e in self.G.edges:
            self.G[e[0]][e[1]]["weight"] = 0.9

        self.f_set = {node: 0.5 for node in self.G.nodes}

    def test_maxBlockFast_single_node(self):
        G = eg.DiGraph()
        G.add_node(0)
        result = eg.maxBlockFast(G, k=1, f_set={0: 1.0}, L=1)
        self.assertEqual(result, [0])

    def test_maxBlockFast_disconnected_graph(self):
        G = eg.DiGraph()
        G.add_nodes_from([0, 1, 2])
        result = eg.maxBlockFast(G, k=2, f_set={0: 0.2, 1: 0.3, 2: 0.5}, L=2)
        self.assertEqual(len(result), 2)

    def test_maxBlock_basic(self):
        result = eg.maxBlock(
            self.G.copy(),
            k=2,
            f_set=self.f_set,
            delta=1,
            eps=0.5,
            c=1,
            flag_weight=True,
        )
        self.assertEqual(len(result), 2)

    def test_maxBlock_unweighted_graph(self):
        G = self.G.copy()
        for e in G.edges:
            del G[e[0]][e[1]]["weight"]
        result = eg.maxBlock(G, k=2, f_set=self.f_set)
        self.assertEqual(len(result), 2)

    def test_maxBlock_random_f_set(self):
        result = eg.maxBlock(self.G.copy(), k=2, f_set=None, flag_weight=True)
        self.assertEqual(len(result), 2)

    def test_maxBlock_invalid_k(self):
        with self.assertRaises(IndexError):
            eg.maxBlock(self.G.copy(), k=100, f_set=self.f_set)


if __name__ == "__main__":
    unittest.main()
