import math
import unittest

import easygraph as eg

from easygraph.utils.exception import EasyGraphError


class test_hypergraph_operation(unittest.TestCase):
    def setUp(self):
        self.g = eg.get_graph_karateclub()
        self.edges = [(1, 2), (8, 4)]
        self.hg = [
            eg.Hypergraph(num_v=10, e_list=self.edges, e_property=None),
            eg.Hypergraph(num_v=2, e_list=[(0, 1)]),
        ]
        # checked -- num_v cannot be set to negative number

    def test_hypergraph_operation(self):
        for i in self.hg:
            print(eg.hypergraph_density(i))
            i.draw(v_color="#e6928f", e_color="#4e9595")

    def test_basic_density(self):
        hg = eg.Hypergraph(num_v=3, e_list=[(0, 1), (1, 2)])
        expected = 2 / (2**3 - 1)
        self.assertAlmostEqual(eg.hypergraph_density(hg), expected)

    def test_density_ignore_singletons(self):
        hg = eg.Hypergraph(num_v=3, e_list=[(0,), (1, 2)])
        expected = 2 / ((2**3 - 1) - 3)
        self.assertAlmostEqual(
            eg.hypergraph_density(hg, ignore_singletons=True), expected
        )

    def test_density_all_singletons(self):
        hg = eg.Hypergraph(num_v=3, e_list=[(0,), (1,), (2,)])
        expected = 3 / (2**3 - 1)
        self.assertAlmostEqual(eg.hypergraph_density(hg), expected)
        expected_ignoring = 3 / ((2**3 - 1) - 3)
        self.assertAlmostEqual(
            eg.hypergraph_density(hg, ignore_singletons=True), expected_ignoring
        )

    def test_no_edges_returns_zero(self):
        hg = eg.Hypergraph(num_v=5, e_list=[])
        self.assertEqual(eg.hypergraph_density(hg), 0.0)

    def test_single_node_single_edge(self):
        hg = eg.Hypergraph(num_v=1, e_list=[(0,)])
        self.assertEqual(eg.hypergraph_density(hg), 1.0)

    def test_density_max_possible_edges(self):
        n = 4
        from itertools import chain
        from itertools import combinations

        powerset = list(
            chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1))
        )
        hg = eg.Hypergraph(num_v=n, e_list=powerset)
        self.assertAlmostEqual(eg.hypergraph_density(hg), 1.0)

    def test_density_zero_division_guard(self):
        # Singleton ignored in n=1 graph should not divide by zero
        hg = eg.Hypergraph(num_v=1, e_list=[(0,)])
        result = eg.hypergraph_density(hg, ignore_singletons=True)
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
