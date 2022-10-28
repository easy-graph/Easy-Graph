import unittest

import easygraph as eg


class EvaluationTest(unittest.TestCase):
    def test_effective_size(self):
        test_graph = eg.Graph([(1, 2), (2, 3), (3, 4)])

        test_nodes_one = [1, 2, 3]
        test_graph_expected_result_one = {1: -1.0, 2: 0.0, 3: 0.0}

        self.assertEqual(
            eg.effective_size(test_graph, test_nodes_one),
            test_graph_expected_result_one,
        )

        test_graph_expected_result_two = {1: -1.0, 4: -1.0}
        test_nodes_two = [1, 4]

        self.assertEqual(
            eg.effective_size(test_graph, test_nodes_two),
            test_graph_expected_result_two,
        )

        test_nodes_three = [5]

        with self.assertRaises(KeyError):
            eg.effective_size(test_graph, test_nodes_three)

        test_four_nodes = None
        test_four_weight = None
        test_four_n_workers = 5

        test_four_expected_result = {1: -1.0, 2: 0.0, 3: 0.0, 4: -1.0}
        self.assertEqual(
            eg.effective_size(
                test_graph, test_four_nodes, test_four_weight, test_four_n_workers
            ),
            test_four_expected_result,
        )

    def test_efficiency(self):
        test_graph = eg.Graph([(1, 2), (2, 3), (3, 4)])

        test_one_nodes = [1, 2, 3]
        test_one_weight = "5"
        test_one_expected_result = {1: 1.0, 2: 1.0, 3: 1.0}

        self.assertEqual(
            eg.efficiency(test_graph, test_one_nodes, test_one_weight),
            test_one_expected_result,
        )

        test_two_nodes = test_one_nodes
        test_two_weight = None
        test_two_expected_result = {1: -1.0, 2: 0.0, 3: 0.0}

        self.assertEqual(
            eg.efficiency(test_graph, test_two_nodes, test_two_weight),
            test_two_expected_result,
        )

        test_three_nodes = [1, 4]
        test_three_weight = None
        test_three_expected_result = {1: -1.0, 4: -1.0}

        self.assertEqual(
            eg.efficiency(test_graph, test_three_nodes, test_three_weight),
            test_three_expected_result,
        )

        test_four_nodes = test_three_nodes
        test_four_weight = "5"
        test_four_expected_result = {1: 1.0, 4: 1.0}

        self.assertEqual(
            eg.efficiency(test_graph, test_four_nodes, test_four_weight),
            test_four_expected_result,
        )

    def test_constraint(self):
        test_one_graph = eg.Graph([(1, 2), (2, 3), (3, 4)])
        test_one_nodes = [1, 2, 4]
        test_one_weight = None
        test_one_n_workers = None
        test_one_expected_result = {1: 1.0, 2: 0.5, 4: 1.0}

        self.assertEqual(
            eg.constraint(
                test_one_graph, test_one_nodes, test_one_weight, test_one_n_workers
            ),
            test_one_expected_result,
        )

        test_two_graph = eg.Graph([(1, 2), (2, 3), (3, 4), (8, 10)])
        test_two_nodes = [1, 4, 10]
        test_two_weight = "4"
        test_two_n_workers = 5
        test_two_expected_result = {4: 1.0, 10: 1.0, 1: 1.0}

        self.assertEqual(
            eg.constraint(
                test_two_graph, test_two_nodes, test_two_weight, test_two_n_workers
            ),
            test_two_expected_result,
        )

    def test_hierarchy(self):
        test_graph = eg.Graph([(1, 2), (2, 3), (3, 4), (8, 10)])
        test_one_expected_result = {1: 0, 2: 0.0, 3: 0.0, 4: 0, 8: 0, 10: 0}

        self.assertEqual(eg.hierarchy(test_graph), test_one_expected_result)

        test_two_n_workers = 5
        test_two_expected_result = {2: 0.0, 8: 0, 4: 0, 1: 0, 10: 0, 3: 0.0}

        self.assertEqual(
            eg.hierarchy(test_graph, n_workers=test_two_n_workers),
            test_two_expected_result,
        )


if __name__ == "__main__":
    unittest.main()
