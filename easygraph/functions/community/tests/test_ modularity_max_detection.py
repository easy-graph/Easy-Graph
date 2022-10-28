import unittest

import easygraph as eg


class ModularityMaxDetectionTest(unittest.TestCase):
    def test_greedy_modularity_communities_graph(self):
        test_graph_one = eg.Graph([(3, 2)])
        test_graph_one_expected_result = [frozenset([2, 3])]
        test_graph_one_actual_result = eg.greedy_modularity_communities(test_graph_one)
        self.assertEqual(test_graph_one_actual_result, test_graph_one_expected_result)

        test_graph_two = eg.Graph([(2, 2)])
        test_graph_two_expected_result = [frozenset([2])]
        test_graph_two_actual_result = eg.greedy_modularity_communities(test_graph_two)
        self.assertEqual(test_graph_two_actual_result, test_graph_two_expected_result)

    def test_greedy_modularity_communities_digraph(self):
        test_graph_three = eg.DiGraph([("A", "B")])
        test_graph_three_actual_result = eg.greedy_modularity_communities(
            test_graph_three
        )
        test_graph_three_expected_result = [frozenset({"A"}), frozenset({"B"})]
        self.assertEqual(
            test_graph_three_actual_result, test_graph_three_expected_result
        )


if __name__ == "__main__":
    unittest.main()
