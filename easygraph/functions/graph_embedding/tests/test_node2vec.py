import unittest

import easygraph as eg
import numpy as np

from easygraph.functions.graph_embedding.NOBE import NOBE
from easygraph.functions.graph_embedding.NOBE import NOBE_GA


class Test_Nobe(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [(1, 4), (2, 4), (4, 1), (0, 4)]
        self.test_graphs = [eg.classes.DiGraph(self.edges)]
        self.test_undirected_graphs = [eg.classes.Graph(self.edges)]
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

        self.valid_graph = eg.Graph([(0, 1), (1, 2), (2, 3)])
        self.directed_graph = eg.DiGraph([(0, 1), (1, 2)])
        self.graph_with_isolated = eg.Graph([(0, 1), (1, 2)])
        self.graph_with_isolated.add_node(5)  # isolated node

    #
    def test_NOBE(self):
        for i in self.test_graphs:
            NOBE(i, K=1)

    def test_NOBE_GA(self):
        for i in self.test_undirected_graphs:
            NOBE_GA(i, K=1)

    def test_nobe_embedding_shape(self):
        emb = NOBE(self.valid_graph, K=2)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (len(self.valid_graph.nodes), 2))

    def test_nobe_ga_embedding_shape(self):
        emb = NOBE_GA(self.valid_graph, K=2)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (len(self.valid_graph.nodes), 2))

    def test_nobe_invalid_k_zero(self):
        emb = NOBE(self.valid_graph, 0)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (len(self.valid_graph), 0))

    def test_nobe_ga_invalid_k_zero(self):
        emb = NOBE_GA(self.valid_graph, 0)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (len(self.valid_graph), 0))

    def test_nobe_with_isolated_node(self):
        emb = NOBE(self.graph_with_isolated, K=2)
        self.assertEqual(emb.shape[0], len(self.graph_with_isolated))


# if __name__ == "__main__":
#     unittest.main()
