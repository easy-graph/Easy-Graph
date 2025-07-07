import unittest

import easygraph as eg

from easygraph.utils.exception import EasyGraphError


class test_hypergraph_operation(unittest.TestCase):
    def setUp(self):
        self.g = eg.get_graph_karateclub()
        self.edges = [(1, 2), (8, 4)]
        self.hg = [
            eg.Hypergraph(num_v=10, e_list=self.edges, e_property=None),
            eg.Hypergraph(num_v=2, e_list=[(0, 1)]),
        ]

    def test_hypergraph_clustering_coefficient(self):
        for i in self.hg:
            print(eg.hypergraph_clustering_coefficient(i))

    def test_hypergraph_local_clustering_coefficient(self):
        for i in self.hg:
            print(eg.hypergraph_local_clustering_coefficient(i))

    def test_hypergraph_two_node_clustering_coefficient(self):
        for i in self.hg:
            print(eg.hypergraph_two_node_clustering_coefficient(i))


class TestHypergraphClustering(unittest.TestCase):
    def setUp(self):
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        self.hg = eg.Hypergraph(num_v=4, e_list=self.edges)

    def test_hypergraph_clustering_coefficient_basic(self):
        cc = eg.hypergraph_clustering_coefficient(self.hg)
        self.assertIsInstance(cc, dict)
        for k, v in cc.items():
            self.assertIn(k, self.hg.v)
            self.assertGreaterEqual(v, 0)

    def test_hypergraph_local_clustering_coefficient_basic(self):
        cc = eg.hypergraph_local_clustering_coefficient(self.hg)
        self.assertIsInstance(cc, dict)
        for k, v in cc.items():
            self.assertIn(k, self.hg.v)
            self.assertGreaterEqual(v, 0)

    def test_hypergraph_two_node_clustering_union(self):
        cc = eg.hypergraph_two_node_clustering_coefficient(self.hg, kind="union")
        self.assertIsInstance(cc, dict)

    def test_hypergraph_two_node_clustering_min(self):
        cc = eg.hypergraph_two_node_clustering_coefficient(self.hg, kind="min")
        self.assertIsInstance(cc, dict)

    def test_hypergraph_two_node_clustering_max(self):
        cc = eg.hypergraph_two_node_clustering_coefficient(self.hg, kind="max")
        self.assertIsInstance(cc, dict)

    def test_hypergraph_two_node_clustering_invalid_kind(self):
        with self.assertRaises(EasyGraphError):
            eg.hypergraph_two_node_clustering_coefficient(self.hg, kind="invalid")

    def test_single_edge(self):
        hg = eg.Hypergraph(num_v=2, e_list=[(0, 1)])
        cc = eg.hypergraph_clustering_coefficient(hg)
        self.assertTrue(all(k in cc for k in hg.v))

    def test_disconnected_nodes(self):
        hg = eg.Hypergraph(num_v=4, e_list=[(0, 1)])
        cc = eg.hypergraph_clustering_coefficient(hg)
        for v in [2, 3]:
            self.assertEqual(cc[v], 0)

    def test_fully_connected_hyperedge(self):
        hg = eg.Hypergraph(num_v=3, e_list=[(0, 1, 2)])
        cc = eg.hypergraph_clustering_coefficient(hg)
        for v in cc.values():
            self.assertEqual(v, 1.0)

    def test_nan_safety_in_two_node_coefficient(self):
        hg = eg.Hypergraph(num_v=1, e_list=[(0,)])
        result = eg.hypergraph_two_node_clustering_coefficient(hg)
        self.assertEqual(result[0], 0.0)


if __name__ == "__main__":
    unittest.main()
