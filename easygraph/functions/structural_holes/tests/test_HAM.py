import unittest

import easygraph as eg


class HAMTest(unittest.TestCase):
    @unittest.skip("eg.get_structural_holes_HAM's return can be random.")
    def test_get_structural_holes_HAM(self):
        test_graph = eg.Graph([(1, 2), (2, 3), (15, 25)])
        test_k = 2
        test_c = 2
        test_ground_truth_labels = [[0], [0], [1], [0], [1]]
        test_expected_result = (
            [25, 15],
            {1: 0, 2: 1, 3: 2, 15: 3, 25: 4},
            {1: 2, 2: 2, 3: 2, 15: 1, 25: 1},
        )

        self.assertEqual(
            eg.get_structural_holes_HAM(
                test_graph, test_k, test_c, test_ground_truth_labels
            ),
            test_expected_result,
        )


if __name__ == "__main__":
    unittest.main()
