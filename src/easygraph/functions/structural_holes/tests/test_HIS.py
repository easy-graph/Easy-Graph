import unittest

import easygraph as eg


class HISTest(unittest.TestCase):
    def test_get_structural_holes_HIS(self):
        test_graph = eg.Graph([(1, 2), (15, 25)])
        test_frozen_set_list = [frozenset([1, 2, 3]), frozenset([4, 5, 6])]
        test_weight = "5"
        test_expected_result = (
            [(0, 1)],
            {1: {0: 1, 1: 0}, 2: {0: 1, 1: 0}, 15: {0: 0, 1: 0}, 25: {0: 0, 1: 0}},
            {1: {0: 0}, 2: {0: 0}, 15: {0: 0}, 25: {0: 0}},
        )

        self.assertEqual(
            eg.get_structural_holes_HIS(
                test_graph, test_frozen_set_list, weight=test_weight
            ),
            test_expected_result,
        )


if __name__ == "__main__":
    unittest.main()
