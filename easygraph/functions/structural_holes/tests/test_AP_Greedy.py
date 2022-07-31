import unittest

import easygraph as eg


class APGreedyTest(unittest.TestCase):

    def test_common_greedy(self):
        test_graph = eg.Graph([(1, 2), (2, 3), (3, 4)])

        self.assertEqual(
            eg.common_greedy(test_graph, 4),
            [3, 2, 4, 1]
        )

        self.assertEqual(
            eg.common_greedy(test_graph, 1),
            [3]
        )

    def test_AP_Greedy(self):
        test_graph = eg.Graph([(1, 2), (2, 3), (3, 10), (15, 25)])

        self.assertEqual(
            eg.AP_Greedy(test_graph, 3),
            [2, 25, 10]
        )

        self.assertEqual(
            eg.AP_Greedy(test_graph, 6, weight='100'),
            [2, 25, 10, 15, 3, 1]
        )


if __name__ == '__main__':
    unittest.main()
