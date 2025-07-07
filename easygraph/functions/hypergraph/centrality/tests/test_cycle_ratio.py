import unittest

import easygraph as eg


class TestCycleRatioCentrality(unittest.TestCase):
    def setUp(self):
        self.G_triangle = eg.Graph()
        self.G_triangle.add_edges([(1, 2), (2, 3), (3, 1)])

        self.G_star = eg.Graph()
        self.G_star.add_edges([(1, 2), (1, 3), (1, 4)])

        self.G_complete = eg.complete_graph(4)

        self.G_disconnected = eg.Graph()
        self.G_disconnected.add_edges([(1, 2), (3, 4)])

    def test_triangle_graph(self):
        result = eg.cycle_ratio_centrality(self.G_triangle.copy())
        self.assertTrue(all(isinstance(v, float) for v in result.values()))
        self.assertEqual(len(result), 3)

    def test_star_graph(self):
        result = eg.cycle_ratio_centrality(self.G_star.copy())
        self.assertEqual(result, {})

    def test_complete_graph(self):
        result = eg.cycle_ratio_centrality(self.G_complete.copy())
        self.assertEqual(len(result), 4)
        self.assertTrue(all(v > 0 for v in result.values()))

    def test_disconnected_graph(self):
        result = eg.cycle_ratio_centrality(self.G_disconnected.copy())
        self.assertEqual(result, {})

    def test_my_all_shortest_paths_valid(self):
        G = eg.Graph()
        G.add_edges([(1, 2), (2, 3), (3, 4)])
        paths = list(eg.my_all_shortest_paths(G, 1, 4))
        self.assertIn([1, 2, 3, 4], paths)

    def test_my_all_shortest_paths_invalid(self):
        G = eg.Graph()
        G.add_edges([(1, 2), (3, 4)])
        with self.assertRaises(eg.EasyGraphNoPath):
            list(eg.my_all_shortest_paths(G, 1, 4))

    def test_getandJudgeSimpleCircle_true(self):
        G = eg.Graph()
        G.add_edges([(1, 2), (2, 3), (3, 1)])
        self.assertTrue(eg.getandJudgeSimpleCircle([1, 2, 3], G))

    def test_getandJudgeSimpleCircle_false(self):
        G = eg.Graph()
        G.add_edges([(1, 2), (2, 3)])
        self.assertFalse(eg.getandJudgeSimpleCircle([1, 2, 3], G))

    def test_statistics_and_calculate_indicators(self):
        SmallestCyclesOfNodes = {1: set(), 2: set(), 3: set()}
        CycLenDict = {3: 0}
        SmallestCycles = {(1, 2, 3)}
        result = eg.StatisticsAndCalculateIndicators(
            SmallestCyclesOfNodes, CycLenDict, SmallestCycles
        )
        self.assertTrue(isinstance(result, dict))
        self.assertIn(1, result)
        self.assertGreater(result[1], 0)


if __name__ == "__main__":
    unittest.main()
