import unittest

import easygraph as eg
import numpy as np


class TestStructuralHoleInfluenceIndex(unittest.TestCase):
    def setUp(self):
        self.G = eg.datasets.get_graph_karateclub()
        self.Com = [
            [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22],
            [9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        ]
        self.valid_seeds = [3, 20, 9]

    def test_ic_model_output(self):
        result = eg.structural_hole_influence_index(
            self.G, self.valid_seeds, self.Com, "IC", seedRatio=0.1, Directed=False
        )
        self.assertIsInstance(result, dict)

    def test_lt_model_output(self):
        result = eg.structural_hole_influence_index(
            self.G, self.valid_seeds, self.Com, "LT", seedRatio=0.1, Directed=False
        )
        self.assertIsInstance(result, dict)

    def test_directed_graph(self):
        DG = self.G.to_directed()
        result = eg.structural_hole_influence_index(
            DG, self.valid_seeds, self.Com, "IC", Directed=True
        )
        self.assertIsInstance(result, dict)

    def test_empty_seed_list(self):
        result = eg.structural_hole_influence_index(self.G, [], self.Com, "IC")
        self.assertEqual(result, {})

    def test_seed_not_in_community(self):
        result = eg.structural_hole_influence_index(self.G, [0], self.Com, "IC")
        self.assertEqual(result, {})

    def test_invalid_model(self):
        with self.assertRaises(Exception):
            eg.structural_hole_influence_index(
                self.G, self.valid_seeds, self.Com, "XYZ"
            )

    def test_empty_community_list(self):
        result = eg.structural_hole_influence_index(self.G, self.valid_seeds, [], "IC")
        self.assertEqual(result, {})

    def test_large_seed_ratio(self):
        result = eg.structural_hole_influence_index(
            self.G, self.valid_seeds, self.Com, "IC", seedRatio=2.0
        )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
