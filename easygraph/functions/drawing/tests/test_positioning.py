import unittest

import easygraph as eg


class PositioningTest(unittest.TestCase):

    def test_random_position(self):
        test_graph = eg.Graph([(2, 6), (6, 10), (15, 25)])
        actual_result_one = eg.random_position(test_graph)

        self.assertListEqual(
            list(actual_result_one.keys()),
            [2, 6, 10, 15, 25]
        )

        for coordinates in actual_result_one.values():
            self.assertEqual(len(coordinates), 2)
            self.assertTrue(0.0 < coordinates[0] < 1.0)
            self.assertTrue(0.0 < coordinates[1] < 1.0)

        actual_result_two = eg.random_position(test_graph, (6, 10))

        for coordinates in actual_result_two.values():
            self.assertTrue(6.0 < coordinates[0] < 7.0)
            self.assertTrue(10.0 < coordinates[1] < 11.0)

    def test_circular_position(self):
        test_graph = eg.Graph([(2, 3), (2, 5), (2, 10)])
        actual_result = eg.circular_position(test_graph, (2, 5))

        self.assertListEqual(
            list(actual_result.keys()),
            [2, 3, 5, 10]
        )

        coordinate_delta = 0.000001
        self.assertAlmostEqual(actual_result.get(2)[1], 5.0, delta=coordinate_delta)
        self.assertAlmostEqual(actual_result.get(3)[0], 2.0, delta=coordinate_delta)
        self.assertAlmostEqual(actual_result.get(5)[1], 5.0, delta=coordinate_delta)
        self.assertAlmostEqual(actual_result.get(10)[0], 2.0, delta=coordinate_delta)

    def test_shell_position(self):
        test_graph = eg.Graph([(1, 6), (1, 5), (4, 10), (4, 5)])
        actual_result_one = eg.shell_position(test_graph, center=(4, 10))

        self.assertListEqual(
            list(actual_result_one.keys()),
            [1, 6, 5, 4, 10]
        )

        for nodes, coordinates in actual_result_one.items():
            self.assertAlmostEqual(coordinates[0], 4, delta=1.0)
            self.assertAlmostEqual(coordinates[1], 10, delta=1.0)

        actual_result_two = eg.shell_position(test_graph, [[4,10]])

        self.assertListEqual(list(actual_result_two.keys()), [4, 10])
        self.assertAlmostEqual(actual_result_two.get(4)[0], 1.0, delta=0.0000001)
        self.assertAlmostEqual(actual_result_two.get(4)[1], 0.0, delta=0.00000005)
        self.assertAlmostEqual(actual_result_two.get(10)[0], -1.0, delta=0.0000001)
        self.assertAlmostEqual(actual_result_two.get(10)[1], -0.0, delta=0.00000005)

    def test_rescale_position(self):
        import numpy as np
        test_position = np.array([[1.0, 10.0], [3.0, 24.0]])
        actual_result = eg.rescale_position(test_position)

        self.assertEqual(actual_result[0][0], -1 * actual_result[1][0])
        self.assertEqual(actual_result[0][1], -1 * actual_result[1][1])
        self.assertEqual(actual_result.shape, (2, 2))

    def test_kamada_kawai_layout(self):
        test_graph = eg.Graph([(2, 6), (2, 10), (15, 25), (15, 30)])
        actual_result = eg.kamada_kawai_layout(test_graph)

        self.assertListEqual(
            list(actual_result.keys()),
            [2, 6, 10, 15, 25, 30]
        )


if __name__ == '__main__':
    unittest.main()
