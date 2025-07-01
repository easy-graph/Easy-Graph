import unittest

import easygraph as eg


class TestHypergraphDegreeCentrality(unittest.TestCase):
    def test_basic_degree_centrality(self):
        hg = eg.Hypergraph(num_v=4, e_list=[(0, 1), (1, 2), (2, 3), (0, 2)])
        result = eg.hyepergraph_degree_centrality(hg)
        expected = {0: 2, 1: 2, 2: 3, 3: 1}
        self.assertEqual(result, expected)

    def test_empty_hypergraph(self):
        hg = eg.Hypergraph(num_v=1, e_list=[])
        result = eg.hyepergraph_degree_centrality(hg)
        self.assertEqual(result, {0: 0})

    def test_single_edge(self):
        hg = eg.Hypergraph(num_v=3, e_list=[(0, 1, 2)])
        result = eg.hyepergraph_degree_centrality(hg)
        expected = {0: 1, 1: 1, 2: 1}
        self.assertEqual(result, expected)

    def test_singleton_nodes(self):
        hg = eg.Hypergraph(num_v=3, e_list=[(0,), (1,), (2,)])
        result = eg.hyepergraph_degree_centrality(hg)
        expected = {0: 1, 1: 1, 2: 1}
        self.assertEqual(result, expected)

    def test_node_with_no_edges(self):
        hg = eg.Hypergraph(num_v=4, e_list=[(0, 1), (1, 2)])
        result = eg.hyepergraph_degree_centrality(hg)
        expected = {0: 1, 1: 2, 2: 1, 3: 0}  # node 3 has no edges
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
