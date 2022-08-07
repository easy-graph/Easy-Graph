import unittest

import easygraph as eg


class ClossnessTest(unittest.TestCase):

    def test_closeness_centrality(self):
        test_graph = eg.Graph([(2, 6), (6, 10), (10, 25), (30, 40), (40, 50)])
        actual_result = eg.closeness_centrality(test_graph)
        self.assertEqual(actual_result.get(2), actual_result.get(25))
        self.assertEqual(actual_result.get(6), actual_result.get(10))
        self.assertEqual(actual_result.get(30), actual_result.get(50))


if __name__ == '__main__':
    unittest.main()
