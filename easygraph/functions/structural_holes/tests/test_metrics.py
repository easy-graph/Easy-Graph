import unittest

import easygraph as eg

from easygraph.functions.structural_holes.metrics import nodes_of_max_cc_without_shs
from easygraph.functions.structural_holes.metrics import structural_hole_influence_index
from easygraph.functions.structural_holes.metrics import sum_of_shortest_paths


class TestStructuralHoleMetrics(unittest.TestCase):
    def setUp(self):
        self.G = eg.datasets.get_graph_karateclub()
        self.shs = [3, 9, 20]
        self.communities = [
            [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22],
            [9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        ]

    def test_sum_of_shortest_paths_valid(self):
        result = sum_of_shortest_paths(self.G, self.shs)
        self.assertIsInstance(result, (int, float))

    def test_sum_of_shortest_paths_empty(self):
        result = sum_of_shortest_paths(self.G, [])
        self.assertEqual(result, 0)

    def test_nodes_of_max_cc_without_shs(self):
        result = nodes_of_max_cc_without_shs(self.G, self.shs)
        self.assertIsInstance(result, int)
        self.assertLessEqual(result, self.G.number_of_nodes())

    def test_nodes_of_max_cc_without_all_nodes(self):
        result = nodes_of_max_cc_without_shs(self.G, list(self.G.nodes))
        self.assertEqual(result, 0)

    def test_structural_hole_influence_index_IC(self):
        result = structural_hole_influence_index(
            self.G,
            self.shs,
            self.communities,
            model="IC",
            Directed=False,
            seedRatio=0.1,
            randSeedIter=2,
            countIterations=5,
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(all(isinstance(k, int) for k in result))
        self.assertTrue(all(isinstance(v, float) for v in result.values()))

    def test_structural_hole_influence_index_LT(self):
        result = structural_hole_influence_index(
            self.G,
            self.shs,
            self.communities,
            model="LT",
            Directed=False,
            seedRatio=0.1,
            randSeedIter=2,
            countIterations=5,
        )
        self.assertIsInstance(result, dict)

    def test_structural_hole_influence_index_variant_LT(self):
        result = structural_hole_influence_index(
            self.G,
            self.shs,
            self.communities,
            model="LT",
            variant=True,
            Directed=False,
            seedRatio=0.1,
            randSeedIter=2,
            countIterations=5,
        )
        self.assertIsInstance(result, dict)

    def test_structural_hole_influence_index_empty_shs(self):
        result = structural_hole_influence_index(
            self.G, [], self.communities, model="IC", Directed=False
        )
        self.assertEqual(result, {})

    def test_structural_hole_influence_index_directed_flag(self):
        result = structural_hole_influence_index(
            self.G,
            self.shs,
            self.communities,
            model="IC",
            Directed=True,
            seedRatio=0.1,
            randSeedIter=2,
            countIterations=5,
        )
        self.assertIsInstance(result, dict)

    def test_structural_hole_influence_index_no_shs_in_any_community(self):
        result = structural_hole_influence_index(
            self.G, [34], self.communities, model="LT", Directed=False
        )
        self.assertIn(34, result)


if __name__ == "__main__":
    unittest.main()
