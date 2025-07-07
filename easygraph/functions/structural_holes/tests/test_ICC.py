import unittest

import easygraph as eg

from easygraph.functions.structural_holes.ICC import AP_BICC
from easygraph.functions.structural_holes.ICC import BICC
from easygraph.functions.structural_holes.ICC import ICC


class TestICCBICCFunctions(unittest.TestCase):
    def setUp(self):
        self.G = eg.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3), (2, 4)])

    def test_icc_basic(self):
        result = ICC(self.G, k=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(node in self.G.nodes for node in result))

    def test_icc_k_exceeds_nodes(self):
        result = ICC(self.G, k=10)
        self.assertLessEqual(len(result), len(self.G.nodes))

    def test_bicc_basic(self):
        result = BICC(self.G, k=2, K=4, l=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(node in self.G.nodes for node in result))

    def test_ap_bicc_basic(self):
        result = AP_BICC(self.G, k=2, K=4, l=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(node in self.G.nodes for node in result))

    def test_icc_disconnected_graph(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        result = ICC(G, k=2)
        self.assertTrue(all(node in G.nodes for node in result))

    def test_bicc_disconnected_graph(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        result = BICC(G, k=1, K=2, l=1)
        self.assertTrue(all(node in G.nodes for node in result))

    def test_ap_bicc_disconnected_graph(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        result = AP_BICC(G, k=1, K=2, l=1)
        self.assertTrue(all(node in G.nodes for node in result))

    def test_icc_single_node(self):
        G = eg.Graph()
        G.add_node(1)
        result = ICC(G, k=1)
        self.assertEqual(result, [1])

    def test_bicc_single_node(self):
        G = eg.Graph()
        G.add_node(1)
        result = BICC(G, k=1, K=1, l=1)
        self.assertEqual(result, [1])

    def test_ap_bicc_single_node(self):
        G = eg.Graph()
        G.add_node(1)
        result = AP_BICC(G, k=1, K=1, l=1)
        self.assertEqual(result, [1])


if __name__ == "__main__":
    unittest.main()
