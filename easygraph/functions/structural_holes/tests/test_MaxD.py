import unittest

import easygraph as eg


class TestStructuralHolesMaxD(unittest.TestCase):
    def setUp(self):
        # Small undirected graph with a bridge between communities
        self.G = eg.Graph()
        self.G.add_edges_from(
            [
                (1, 2),
                (2, 3),  # Community A
                (4, 5),
                (5, 6),  # Community B
                (3, 4),  # Bridge edge
            ]
        )
        self.communities = [frozenset([1, 2, 3]), frozenset([4, 5, 6])]

    def test_basic_top1(self):
        result = eg.get_structural_holes_MaxD(self.G, k=1, C=self.communities)
        self.assertEqual(len(result), 1)
        self.assertIn(result[0], self.G.nodes)

    def test_top_k_greater_than_1(self):
        result = eg.get_structural_holes_MaxD(self.G, k=3, C=self.communities)
        self.assertEqual(len(result), 3)
        for node in result:
            self.assertIn(node, self.G.nodes)

    def test_unweighted_graph(self):
        result = eg.get_structural_holes_MaxD(self.G, k=2, C=self.communities)
        self.assertEqual(len(result), 2)

    def test_disconnected_communities(self):
        self.G.add_node(7)
        new_comms = [frozenset([1, 2]), frozenset([7])]
        result = eg.get_structural_holes_MaxD(self.G, k=1, C=new_comms)
        self.assertTrue(all(node in self.G.nodes for node in result))

    def test_single_node_communities(self):
        result = eg.get_structural_holes_MaxD(
            self.G, k=1, C=[frozenset([1]), frozenset([6])]
        )
        self.assertTrue(all(node in self.G.nodes for node in result))

    def test_disconnected_graph(self):
        G = eg.Graph()
        G.add_nodes_from([1, 2, 3, 4])
        G.add_edges_from([(1, 2), (3, 4)])
        comms = [frozenset([1, 2]), frozenset([3, 4])]
        result = eg.get_structural_holes_MaxD(G, k=2, C=comms)
        self.assertEqual(len(result), 2)

    def test_duplicate_nodes_in_communities(self):
        result = eg.get_structural_holes_MaxD(
            self.G, k=2, C=[frozenset([1, 2, 3]), frozenset([3, 4, 5])]
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
