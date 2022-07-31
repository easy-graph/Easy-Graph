import unittest

import easygraph as eg


class FlowBetweennessCentralityTest(unittest.TestCase):

    def test_flowbetweenness_centrality(self):
        test_graph = eg.DiGraph(
            [
                (4, 5),
                (5, 9),
                (9, 10),
                (9, 11),
                (12, 14)
            ]
        )
        actual_result = eg.flowbetweenness_centrality(test_graph)
        self.assertEqual(actual_result.get(5), 3.0)
        self.assertEqual(actual_result.get(9), 4.0)
        for i in [4, 10, 11, 12, 14]:
            self.assertEqual(actual_result.get(i), 0.0)

        test_graph = eg.DiGraph(
            [
                (4, 5),
                (5, 8),
                (8, 10),
                (12, 14)
            ]
        )
        actual_result = eg.flowbetweenness_centrality(test_graph)
        for i in [5, 8]:
            self.assertEqual(actual_result.get(i), 2.0)

        for i in [4, 10, 12, 14]:
            self.assertEqual(actual_result.get(i), 0.0)


if __name__ == '__main__':
    unittest.main()
