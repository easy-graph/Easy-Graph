import unittest

import easygraph as eg
import numpy as np

from easygraph.exception import EasyGraphError


class TestVectorCentrality(unittest.TestCase):
    def test_single_edge(self):
        hg = eg.Hypergraph(num_v=3, e_list=[(0, 1, 2)])
        result = eg.vector_centrality(hg)
        self.assertEqual(set(result.keys()), {0, 1, 2})
        for val in result.values():
            self.assertEqual(len(val), 2)  # because D = 3 → k = 2 and 3

    def test_multiple_edges_different_orders(self):
        hg = eg.Hypergraph(num_v=4, e_list=[(0, 1), (1, 2, 3)])
        result = eg.vector_centrality(hg)
        self.assertEqual(set(result.keys()), {0, 1, 2, 3})
        for val in result.values():
            self.assertEqual(len(val), 2)
            self.assertTrue(all(isinstance(x, (float, np.floating)) for x in val))

    def test_disconnected_hypergraph_raises(self):
        hg = eg.Hypergraph(num_v=6, e_list=[(0, 1), (2, 3)])
        with self.assertRaises(EasyGraphError):
            eg.vector_centrality(hg)

    def test_non_consecutive_node_ids(self):
        hg = eg.Hypergraph(num_v=5, e_list=[(0, 2, 4)])
        result = eg.vector_centrality(hg)
        self.assertEqual(len(result), 5)
        for val in result.values():
            self.assertEqual(len(val), 2)

    def test_index_error_due_to_wrong_num_v(self):
        with self.assertRaises(eg.EasyGraphError):
            eg.Hypergraph(num_v=3, e_list=[(0, 1, 5)])


if __name__ == "__main__":
    unittest.main()
