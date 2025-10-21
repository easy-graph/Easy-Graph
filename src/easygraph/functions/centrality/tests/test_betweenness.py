import unittest

import easygraph as eg


class BetweennessTest(unittest.TestCase):
    def test_betweenness_centrality(self):
        test_graph = eg.Graph([(2, 6), (6, 10), (10, 25), (30, 40), (40, 50)])
        actual_result = eg.betweenness_centrality(test_graph, normalized=False)

        self.assertEqual(actual_result.get(6), 2.0)
        self.assertEqual(actual_result.get(10), 2.0)
        self.assertEqual(actual_result.get(40), 1.0)

        for i in [2, 25, 30, 50]:
            self.assertEqual(actual_result.get(i), 0.0)


if __name__ == "__main__":
    unittest.main()
