import unittest

import easygraph as eg


class TestStructuralHoleSpanners(unittest.TestCase):
    def setUp(self):
        self.G = eg.get_graph_karateclub()
        self.small_graph = eg.Graph()
        self.small_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        self.disconnected_graph = eg.Graph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3)])

    def test_common_greedy_basic(self):
        result = eg.common_greedy(self.G, k=3)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for node in result:
            self.assertIn(node, self.G.nodes)

    def test_ap_greedy_basic(self):
        result = eg.AP_Greedy(self.G, k=3)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for node in result:
            self.assertIn(node, self.G.nodes)

    def test_common_greedy_k_zero(self):
        result = eg.common_greedy(self.G, k=0)
        self.assertEqual(result, [])

    def test_ap_greedy_k_zero(self):
        result = eg.AP_Greedy(self.G, k=0)
        self.assertEqual(result, [])

    def test_common_greedy_on_disconnected_graph(self):
        result = eg.common_greedy(self.disconnected_graph, k=2)
        self.assertEqual(len(result), 2)
        for node in result:
            self.assertIn(node, self.disconnected_graph.nodes)

    def test_ap_greedy_on_disconnected_graph(self):
        result = eg.AP_Greedy(self.disconnected_graph, k=2)
        self.assertEqual(len(result), 2)
        for node in result:
            self.assertIn(node, self.disconnected_graph.nodes)

    def test_common_greedy_with_custom_c(self):
        result_default = eg.common_greedy(self.G, k=2)
        result_custom = eg.common_greedy(self.G, k=2, c=2.5)
        self.assertEqual(len(result_default), 2)
        self.assertEqual(len(result_custom), 2)

    def test_ap_greedy_with_custom_c(self):
        result_default = eg.AP_Greedy(self.G, k=2)
        result_custom = eg.AP_Greedy(self.G, k=2, c=2.5)
        self.assertEqual(len(result_default), 2)
        self.assertEqual(len(result_custom), 2)

    def test_common_greedy_unweighted_vs_weighted(self):
        # With and without weights
        G_weighted = self.small_graph.copy()
        for edge in G_weighted.edges:
            u, v = edge[:2]
            G_weighted[u][v]["weight"] = 1.0

        result_unweighted = eg.common_greedy(G_weighted, k=2, weight=None)
        result_weighted = eg.common_greedy(G_weighted, k=2, weight="weight")
        self.assertEqual(len(result_unweighted), 2)
        self.assertEqual(len(result_weighted), 2)

    def test_ap_greedy_unweighted_vs_weighted(self):
        G_weighted = self.small_graph.copy()
        for edge in G_weighted.edges:
            u, v = edge[:2]
            G_weighted[u][v]["weight"] = 1.0

        result_unweighted = eg.AP_Greedy(G_weighted, k=2, weight=None)
        result_weighted = eg.AP_Greedy(G_weighted, k=2, weight="weight")
        self.assertEqual(len(result_unweighted), 2)
        self.assertEqual(len(result_weighted), 2)


if __name__ == "__main__":
    unittest.main()
