import unittest

import easygraph as eg


class test_random_network(unittest.TestCase):
    def setUp(self):
        self.G = eg.datasets.get_graph_karateclub()

    def test_erdos_renyi_M(self):
        print(eg.erdos_renyi_M(8, 5).edges)

    def test_erdos_renyi_P(self):
        print(eg.erdos_renyi_P(8, 0.2).nodes)

    def test_fast_erdos_renyi_P(self):
        print(eg.fast_erdos_renyi_P(8, 0.2).nodes)

    def test_WS_Random(self):
        print(eg.WS_Random(8, 1, 0.5).nodes)

    def test_graph_Gnm(self):
        print(eg.graph_Gnm(8, 5).nodes)

    def test_erdos_renyi_M_max_edges(self):
        n = 5
        max_edges = n * (n - 1) // 2
        G = eg.erdos_renyi_M(n, max_edges)
        self.assertEqual(len(G.edges), max_edges)

    def test_erdos_renyi_P_extreme_p(self):
        G0 = eg.erdos_renyi_P(10, 0.0)
        G1 = eg.erdos_renyi_P(10, 1.0)
        self.assertEqual(len(G0.edges), 0)
        self.assertEqual(len(G1.edges), 45)  # 10 * 9 / 2

    def test_fast_erdos_renyi_P_large_p(self):
        G = eg.fast_erdos_renyi_P(10, 0.9)
        self.assertEqual(len(G.nodes), 10)

    def test_WS_Random_structure(self):
        G = eg.WS_Random(10, 2, 0.1)
        self.assertEqual(len(G.nodes), 10)
        self.assertTrue(all(0 <= u < 10 and 0 <= v < 10 for u, v, *_ in G.edges))

    def test_WS_Random_invalid_k(self):
        G = eg.WS_Random(5, 5, 0.1)
        self.assertIsNone(G)

    def test_graph_Gnm_basic(self):
        G = eg.graph_Gnm(10, 15)
        self.assertEqual(len(G.nodes), 10)
        self.assertEqual(len(G.edges), 15)

    def test_graph_Gnm_invalid_inputs(self):
        with self.assertRaises(AssertionError):
            eg.graph_Gnm(1, 1)
        with self.assertRaises(AssertionError):
            eg.graph_Gnm(5, 11)  # 5*4/2 = 10 max


if __name__ == "__main__":
    unittest.main()
