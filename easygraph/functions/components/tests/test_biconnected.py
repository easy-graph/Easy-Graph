import unittest

import easygraph as eg


class BiConnectedTest(unittest.TestCase):
    def test_is_biconnected(self):
        test_graph = eg.DiGraph(
            [
                (4, 5),
                (5, 6),
                (6, 10),
                (10, 12),
                (12, 14),
            ]
        )
        self.assertEqual(eg.is_biconnected(test_graph), False)

        test_graph = eg.DiGraph(
            [
                (4, 5),
                (5, 6),
                (6, 10),
                (10, 12),
                (12, 4),
            ]
        )
        self.assertEqual(eg.is_biconnected(test_graph), True)

    def test_biconnected_components(self):
        test_graph = eg.DiGraph([(4, 5), (5, 6), (6, 4), (20, 30), (30, 20)])
        actual_result = eg.biconnected_components(test_graph)
        self.assertListEqual(actual_result[0], [(4, 5), (5, 6), (6, 4)])
        self.assertListEqual(actual_result[1], [(20, 30)])

    def test_generator_biconnected_components_nodes(self):
        test_graph = eg.DiGraph([(4, 5), (5, 6), (6, 4), (20, 30), (30, 40)])
        actual_result = [
            _ for _ in eg.generator_biconnected_components_nodes(test_graph)
        ]
        self.assertListEqual(actual_result, [{4, 5, 6}, {40, 30}, {20, 30}])

    def test_generator_biconnected_components_edges(self):
        test_graph = eg.DiGraph([(4, 5), (5, 6), (6, 4), (20, 30)])
        actual_result = [
            _ for _ in eg.generator_biconnected_components_edges(test_graph)
        ]
        self.assertListEqual(actual_result, [[(4, 5), (5, 6), (6, 4)], [(20, 30)]])

    def test_generator_articulation_points(self):
        test_graph = eg.DiGraph(
            [(4, 5), (6, 10), (10, 20), (12, 15), (15, 30), (25, 40)]
        )
        actual_result = [_ for _ in eg.generator_articulation_points(test_graph)]
        self.assertListEqual(actual_result, [10, 15])


if __name__ == "__main__":
    unittest.main()
