import unittest

import easygraph as eg
import numpy as np


class TestHAMStructuralHoles(unittest.TestCase):
    def setUp(self):
        self.G = eg.Graph()
        self.G.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (1, 2),  # Community 0
                (3, 4),
                (3, 5),
                (4, 5),  # Community 1
                (2, 3),  # Bridge between 0 and 1
                (6, 7),
                (6, 8),
                (7, 8),  # Community 2
            ]
        )
        self.labels = [[0], [0], [0], [1], [1], [1], [2], [2], [2]]

    def test_output_structure(self):
        top_k, sh_score, cmnt_labels = eg.get_structural_holes_HAM(
            self.G, k=2, c=3, ground_truth_labels=self.labels
        )
        self.assertIsInstance(top_k, list)
        self.assertTrue(all(isinstance(n, int) for n in top_k))
        self.assertEqual(len(top_k), 2)

        self.assertIsInstance(sh_score, dict)
        self.assertEqual(len(sh_score), self.G.number_of_nodes())
        self.assertTrue(
            all(isinstance(k, int) and isinstance(v, int) for k, v in sh_score.items())
        )

        self.assertIsInstance(cmnt_labels, dict)
        self.assertEqual(len(cmnt_labels), self.G.number_of_nodes())

    def test_single_community(self):
        labels = [[0]] * self.G.number_of_nodes()
        top_k, _, _ = eg.get_structural_holes_HAM(
            self.G, k=1, c=1, ground_truth_labels=labels
        )
        self.assertEqual(len(top_k), 1)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            eg.get_structural_holes_HAM(
                self.G, k=-1, c=2, ground_truth_labels=self.labels
            )

    def test_invalid_c(self):
        with self.assertRaises(ValueError):
            eg.get_structural_holes_HAM(
                self.G, k=2, c=0, ground_truth_labels=self.labels
            )

    def test_mismatched_labels(self):
        bad_labels = [[0]] * (self.G.number_of_nodes() - 1)
        with self.assertRaises(ValueError):
            eg.get_structural_holes_HAM(
                self.G, k=2, c=2, ground_truth_labels=bad_labels
            )


if __name__ == "__main__":
    unittest.main()
