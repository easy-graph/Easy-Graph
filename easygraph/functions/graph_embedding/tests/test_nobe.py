import unittest

import easygraph as eg
import easygraph.functions.graph_embedding as fn
import numpy as np


class Test_Nobe(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [(1, 4), (2, 4), (4, 1), (0, 4), (4, 3)]
        self.test_directed_graphs = [eg.DiGraph()]
        self.test_undirected_graphs = [eg.Graph(self.edges)]
        self.test_directed_graphs.append(eg.classes.DiGraph(self.edges))
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

        self.valid_graph = eg.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
        self.directed_graph = eg.DiGraph([(0, 1), (1, 2)])
        self.graph_with_isolated = eg.Graph()
        self.graph_with_isolated.add_edges_from([(0, 1), (1, 2)])
        self.graph_with_isolated.add_node(3)
        self.graph_with_isolated.add_node(4)

    def test_NOBE(self):
        fn.NOBE(self.test_undirected_graphs[0], 1)

    def test_NOBE_GA(self):
        """
        for i in self.test_graphs:
            eg.functions.NOBE_GA(i, K=1)
            print(i)
        """
        fn.NOBE_GA(self.test_directed_graphs[1], 1)

    def test_nobe_output_shape(self):
        emb = fn.NOBE(self.valid_graph, K=2)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape[1], 2)

    def test_nobe_ga_output_shape(self):
        undirected_graph = eg.Graph([(0, 1), (1, 2), (2, 3)])
        emb = fn.NOBE_GA(undirected_graph, K=2)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape[1], 2)

    def test_nobe_on_graph_with_isolated_nodes(self):
        emb = fn.NOBE(self.graph_with_isolated, K=2)
        self.assertEqual(emb.shape[0], len(self.graph_with_isolated))

    def test_nobe_invalid_K_zero(self):
        emb = fn.NOBE(self.valid_graph, 0)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (len(self.valid_graph), 0))


if __name__ == "__main__":
    unittest.main()
