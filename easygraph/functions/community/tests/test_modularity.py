import unittest

import easygraph as eg


class TestModularity(unittest.TestCase):
    def setUp(self):
        self.G = eg.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        self.DG = eg.DiGraph()
        self.DG.add_edges_from([(0, 1), (1, 2), (2, 0)])

        self.G_weighted = eg.Graph()
        self.G_weighted.add_edge(0, 1, weight=2)
        self.G_weighted.add_edge(1, 2, weight=3)
        self.G_weighted.add_edge(2, 0, weight=1)

        self.G_selfloop = eg.Graph()
        self.G_selfloop.add_edges_from([(0, 0), (1, 1), (0, 1)])

        self.G_empty = eg.Graph()

    def test_undirected_modularity(self):
        communities = [{0, 1}, {2, 3}]
        q = eg.functions.community.modularity(self.G, communities)
        self.assertIsInstance(q, float)

    def test_directed_modularity(self):
        communities = [{0, 1, 2}]
        q = eg.functions.community.modularity(self.DG, communities)
        self.assertIsInstance(q, float)

    def test_weighted_graph(self):
        communities = [{0, 1}, {2}]
        q = eg.functions.community.modularity(
            self.G_weighted, communities, weight="weight"
        )
        self.assertIsInstance(q, float)

    def test_self_loops(self):
        communities = [{0, 1}]
        q = eg.functions.community.modularity(self.G_selfloop, communities)
        self.assertIsInstance(q, float)

    def test_single_community(self):
        communities = [{0, 1, 2, 3}]
        q = eg.functions.community.modularity(self.G, communities)
        self.assertIsInstance(q, float)

    def test_each_node_its_own_community(self):
        communities = [{0}, {1}, {2}, {3}]
        q = eg.functions.community.modularity(self.G, communities)
        self.assertIsInstance(q, float)

    def test_empty_graph(self):
        with self.assertRaises(ZeroDivisionError):
            eg.functions.community.modularity(self.G_empty, [])

    def test_empty_community_list(self):
        q = eg.functions.community.modularity(self.G, [])
        self.assertEqual(q, 0.0)

    def test_non_list_communities(self):
        communities = (set([0, 1]), set([2, 3]))
        q = eg.functions.community.modularity(self.G, communities)
        self.assertIsInstance(q, float)


if __name__ == "__main__":
    unittest.main()
