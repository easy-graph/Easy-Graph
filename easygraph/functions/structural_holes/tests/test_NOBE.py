import unittest

import easygraph as eg


class TestNOBESpanners(unittest.TestCase):
    def setUp(self):
        self.G = eg.datasets.get_graph_karateclub()

    def test_nobe_sh_basic(self):
        result = eg.NOBE_SH(self.G, K=2, topk=3)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(node in self.G.nodes for node in result))

    def test_nobe_ga_sh_basic(self):
        result = eg.NOBE_GA_SH(self.G, K=2, topk=3)
        self.assertEqual(len(result), 3)

    def test_nobe_sh_topk_equals_n(self):
        result = eg.NOBE_SH(self.G, K=2, topk=self.G.number_of_nodes())
        self.assertEqual(len(result), self.G.number_of_nodes())

    def test_nobe_ga_sh_topk_greater_than_n(self):
        result = eg.NOBE_GA_SH(self.G, K=2, topk=self.G.number_of_nodes() + 5)
        self.assertEqual(len(result), self.G.number_of_nodes())

    def test_nobe_sh_k_equals_1(self):
        result = eg.NOBE_SH(self.G, K=1, topk=2)
        self.assertEqual(len(result), 2)

    def test_nobe_ga_sh_k_equals_1(self):
        result = eg.NOBE_GA_SH(self.G, K=1, topk=2)
        self.assertEqual(len(result), 2)

    def test_nobe_sh_disconnected_graph(self):
        G = eg.Graph()
        G.add_edges_from([(1, 2), (3, 4)])
        result = eg.NOBE_SH(G, K=2, topk=2)
        self.assertEqual(len(result), 2)

    def test_nobe_ga_sh_disconnected_graph(self):
        G = eg.Graph()
        G.add_edges_from([(1, 2), (3, 4)])
        result = eg.NOBE_GA_SH(G, K=2, topk=2)
        self.assertEqual(len(result), 2)

    def test_nobe_sh_empty_graph(self):
        G = eg.Graph()
        with self.assertRaises(ValueError):
            eg.NOBE_SH(G, K=2, topk=1)

    def test_nobe_ga_sh_empty_graph(self):
        G = eg.Graph()
        with self.assertRaises(ValueError):
            eg.NOBE_GA_SH(G, K=2, topk=1)

    def test_nobe_sh_invalid_k(self):
        with self.assertRaises(ValueError):
            eg.NOBE_SH(self.G, K=0, topk=3)

    def test_nobe_ga_sh_invalid_k(self):
        with self.assertRaises(ValueError):
            eg.NOBE_GA_SH(self.G, K=0, topk=3)

    def test_nobe_sh_invalid_topk(self):
        with self.assertRaises(ValueError):
            eg.NOBE_SH(self.G, K=2, topk=0)

    def test_nobe_ga_sh_invalid_topk(self):
        with self.assertRaises(ValueError):
            eg.NOBE_GA_SH(self.G, K=2, topk=0)


if __name__ == "__main__":
    unittest.main()
