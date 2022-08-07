import unittest

import easygraph as eg


class MSTTest(unittest.TestCase):
    def test_minimum_spanning_edges(self):
        test_one_graph = eg.Graph([(2, 6), (6, 10), (15, 25), (15, 25)])
        actual_result = [_ for _ in eg.minimum_spanning_edges(test_one_graph)]
        self.assertEqual(actual_result, [(2, 6, {}), (6, 10, {}), (15, 25, {})])

    def test_maximum_spanning_tree(self):
        test_one_graph = eg.Graph()
        test_one_graph.add_edge(0, 3, weight=2)
        test_one_graph.add_edge(2, 6, weight=3)
        print(test_one_graph.edges)
        actual_result = [_ for _ in eg.maximum_spanning_tree(test_one_graph)]
        self.assertEqual([0, 3, 2, 6], actual_result)


if __name__ == '__main__':
    unittest.main()
