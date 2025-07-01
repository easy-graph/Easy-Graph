import unittest

import easygraph as eg
import numpy as np


class TestHypergraphSCentrality(unittest.TestCase):
    def setUp(self):
        # Simple test hypergraph
        self.hg = eg.Hypergraph(num_v=5, e_list=[(0, 1), (1, 2, 3), (3, 4)])
        self.empty_hg = eg.Hypergraph(num_v=1, e_list=[])
        self.singleton_hg = eg.Hypergraph(num_v=3, e_list=[(0,), (1,), (2,)])

    def test_s_betweenness_normal(self):
        result = eg.s_betweenness(self.hg)
        self.assertIsInstance(result, (list, dict))
        self.assertTrue(all(isinstance(x, (int, float)) for x in result))

    def test_s_closeness_normal(self):
        result = eg.s_closeness(self.hg)
        self.assertIsInstance(result, (list, dict))
        self.assertTrue(all(isinstance(x, (int, float)) for x in result))

    def test_s_eccentricity_all(self):
        result = eg.s_eccentricity(self.hg)
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertIsInstance(v, (int, float, np.integer, np.floating))

    def test_s_eccentricity_edges_false(self):
        result = eg.s_eccentricity(self.hg, edges=False)
        self.assertIsInstance(result, dict)

    def test_s_eccentricity_invalid_source(self):
        with self.assertRaises(KeyError):
            eg.s_eccentricity(self.hg, source=(999, 888))


if __name__ == "__main__":
    unittest.main()
