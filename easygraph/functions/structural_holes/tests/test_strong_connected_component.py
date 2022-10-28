import unittest

import easygraph as eg


class StronglyConnectedComponentTest(unittest.TestCase):
    def test_strongly_connected_components(self):
        test_graph = eg.DiGraph([(4, 6), (6, 8), (8, 10), (12, 14)])

        actual_result = []
        for result in eg.strongly_connected_components(test_graph):
            actual_result.append(result)
        expected_result = [{10}, {8}, {6}, {4}, {14}, {12}]

        self.assertEqual(actual_result, expected_result)

    def test_number_strongly_connected_components(self):
        test_graph = eg.DiGraph(
            [
                (4, 6),
                (100, 10),
                (8, 10),
                (12, 14),
                (10, 100),
            ]
        )
        self.assertEqual(eg.number_strongly_connected_components(test_graph), 6)

    def test_condensation(self):
        test_graph = eg.DiGraph(
            [
                (4, 6),
                (8, 10),
                (10, 100),
            ]
        )

        self.assertEqual(
            eg.condensation(test_graph).edges, [(1, 0, {}), (3, 2, {}), (4, 3, {})]
        )


if __name__ == "__main__":
    unittest.main()
