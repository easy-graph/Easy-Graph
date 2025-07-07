import unittest

import easygraph as eg

from easygraph.functions.structural_holes.weakTie import weakTie
from easygraph.functions.structural_holes.weakTie import weakTieLocal


class TestWeakTieFunctions(unittest.TestCase):
    def setUp(self):
        self.G = eg.DiGraph()
        self.G.add_edges_from(
            [
                (1, 5),
                (1, 4),
                (2, 1),
                (2, 6),
                (2, 9),
                (3, 4),
                (3, 1),
                (4, 3),
                (4, 1),
                (4, 5),
                (5, 4),
                (5, 8),
                (6, 1),
                (6, 2),
                (7, 2),
                (7, 3),
                (7, 10),
                (8, 4),
                (8, 5),
                (9, 6),
                (9, 10),
                (10, 7),
                (10, 9),
            ]
        )
        self.threshold = 0.2
        self.k = 3

    def test_weak_tie_returns_top_k(self):
        SHS_list, score_dict = weakTie(self.G.copy(), self.threshold, self.k)
        self.assertEqual(len(SHS_list), self.k)
        self.assertTrue(all(node in self.G.nodes for node in SHS_list))

    def test_weak_tie_zero_k(self):
        SHS_list, _ = weakTie(self.G.copy(), self.threshold, 0)
        self.assertEqual(SHS_list, [])

    def test_with_isolated_node(self):
        self.G.add_node(99)
        SHS_list, score_dict = weakTie(self.G.copy(), self.threshold, self.k)
        self.assertIn(99, score_dict)
        self.assertIsInstance(score_dict[99], (int, float))


if __name__ == "__main__":
    unittest.main()
