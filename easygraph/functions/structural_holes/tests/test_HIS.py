import unittest

import easygraph as eg

from easygraph.functions.structural_holes import get_structural_holes_HIS


class TestHISStructuralHoles(unittest.TestCase):
    def setUp(self):
        self.G = eg.Graph()
        self.G.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),  # Community 0
                (3, 4),
                (4, 5),
                (5, 3),  # Community 1
                (2, 3),  # Bridge between communities
            ]
        )
        self.communities = [frozenset([0, 1, 2]), frozenset([3, 4, 5])]

    def test_normal_output_structure(self):
        S, I, H = get_structural_holes_HIS(self.G, self.communities)
        self.assertIsInstance(S, list)
        self.assertTrue(all(isinstance(s, tuple) for s in S))
        self.assertIsInstance(I, dict)
        self.assertIsInstance(H, dict)
        self.assertEqual(set(I.keys()), set(self.G.nodes))
        self.assertEqual(set(H.keys()), set(self.G.nodes))
        self.assertTrue(all(isinstance(v, dict) for v in I.values()))
        self.assertTrue(all(isinstance(v, dict) for v in H.values()))

    def test_empty_graph(self):
        G = eg.Graph()
        communities = []
        S, I, H = get_structural_holes_HIS(G, communities)
        self.assertEqual(S, [])
        self.assertEqual(I, {})
        self.assertEqual(H, {})

    def test_single_node_community(self):
        G = eg.Graph()
        G.add_node(42)
        communities = [frozenset([42])]
        S, I, H = get_structural_holes_HIS(G, communities)
        self.assertEqual(list(I.keys()), [42])
        self.assertEqual(list(H.keys()), [42])
        self.assertEqual(list(I[42].values())[0], 0)

    def test_disconnected_communities(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 5)])
        communities = [frozenset([0, 1, 2]), frozenset([3, 4, 5])]
        S, I, H = get_structural_holes_HIS(G, communities)
        self.assertEqual(set(I.keys()), set(G.nodes))

    def test_node_in_multiple_communities(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        communities = [frozenset([0, 1]), frozenset([1, 2]), frozenset([2, 3])]
        S, I, H = get_structural_holes_HIS(G, communities)
        self.assertIn(1, I)
        self.assertGreaterEqual(len(I[1]), 2)

    def test_weighted_graph(self):
        G = eg.Graph()
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(1, 2, weight=3.0)
        G.add_edge(2, 0, weight=1.0)
        G.add_edge(2, 3, weight=4.0)
        G.add_edge(3, 4, weight=5.0)
        G.add_edge(4, 5, weight=6.0)
        G.add_edge(5, 3, weight=1.0)
        communities = [frozenset([0, 1, 2]), frozenset([3, 4, 5])]
        S, I, H = get_structural_holes_HIS(G, communities, weight="weight")
        self.assertIsInstance(list(I[0].values())[0], float)

    def test_convergence_with_high_epsilon(self):
        S, I, H = get_structural_holes_HIS(self.G, self.communities, epsilon=1.0)
        self.assertTrue(S)
        self.assertEqual(set(I.keys()), set(self.G.nodes))


if __name__ == "__main__":
    unittest.main()
