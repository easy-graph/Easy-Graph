import math
import unittest

import easygraph as eg

from easygraph.functions.structural_holes import constraint
from easygraph.functions.structural_holes import effective_size
from easygraph.functions.structural_holes import efficiency
from easygraph.functions.structural_holes import hierarchy


class TestStructuralHoleMetrics(unittest.TestCase):
    def setUp(self):
        self.G = eg.Graph()
        self.G.add_edges_from(
            [
                (0, 1, {"weight": 1.0}),
                (0, 2, {"weight": 2.0}),
                (1, 2, {"weight": 1.0}),
                (2, 3, {"weight": 3.0}),
                (3, 4, {"weight": 1.0}),
            ]
        )
        self.G.add_node(5)  # isolated node

    def test_effective_size_unweighted(self):
        result = effective_size(self.G)
        self.assertIn(0, result)
        self.assertTrue(math.isnan(result[5]))

    def test_effective_size_weighted(self):
        result = effective_size(self.G, weight="weight")
        self.assertIn(0, result)
        self.assertTrue(math.isnan(result[5]))

    def test_constraint_unweighted(self):
        result = constraint(self.G)
        self.assertIn(0, result)
        self.assertTrue(math.isnan(result[5]))

    def test_constraint_weighted(self):
        result = constraint(self.G, weight="weight")
        self.assertIn(0, result)
        self.assertTrue(math.isnan(result[5]))

    def test_hierarchy_unweighted(self):
        result = hierarchy(self.G)
        self.assertIn(0, result)
        self.assertEqual(result[5], 0)

    def test_hierarchy_weighted(self):
        result = hierarchy(self.G, weight="weight")
        self.assertIn(0, result)
        self.assertEqual(result[5], 0)

    def test_disconnected_components(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # 2 components
        for func in [effective_size, efficiency, constraint, hierarchy]:
            result = func(G)
            self.assertEqual(set(result.keys()), set(G.nodes))

    def test_directed_graph_support(self):
        DG = eg.DiGraph()
        DG.add_edges_from([(0, 1), (1, 2)])
        result = effective_size(DG)
        self.assertIsInstance(result, dict)
        self.assertTrue(all(node in result for node in DG.nodes))


if __name__ == "__main__":
    unittest.main()
