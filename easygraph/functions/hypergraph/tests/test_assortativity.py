import math
import unittest

from itertools import combinations

import easygraph as eg
import numpy as np

from easygraph.utils.exception import EasyGraphError


class test_assortativity(unittest.TestCase):
    def setUp(self):
        self.g = eg.get_graph_karateclub()
        self.edges = [(8, 9), (1, 2), (8, 4), (3, 6), (1, 3), (6, 4)]
        self.hg = [
            eg.Hypergraph(num_v=10, e_list=self.edges, e_property=None),
            eg.Hypergraph(num_v=2, e_list=[(0, 1)]),
        ]
        # Valid uniform hypergraph
        self.hg_uniform = eg.Hypergraph(
            num_v=5,
            e_list=[
                (0, 1, 2),
                (1, 2, 3),
                (2, 3, 4),
            ],
        )

        # Non-uniform hypergraph
        self.hg_non_uniform = eg.Hypergraph(
            num_v=4,
            e_list=[
                (0, 1),
                (2, 3, 0),
            ],
        )

        # Singleton edge hypergraph (still needs num_v > 0)
        self.hg_singleton = eg.Hypergraph(
            num_v=3,
            e_list=[
                (0,),
                (1, 2),
            ],
        )

        # "Empty" hypergraph (has 1 node but no edges)
        self.hg_empty = eg.Hypergraph(
            num_v=1,
            e_list=[],
        )

    def test_dynamical_assortativity(self):
        for i in self.hg:
            degs = i.deg_v
            print(degs)
            k1 = sum(degs) / len(degs)
            print(k1)
            k2 = np.mean(np.array(degs) ** 2)
            print(k2)
            kk1 = np.mean(
                [degs[n1] * degs[n2] for e in i.e[0] for n1, n2 in combinations(e, 2)]
            )
            print(kk1)
            print(eg.dynamical_assortativity(i))
            print()

    def test_degree_assortativity(self):
        for i in self.hg:
            print(eg.degree_assortativity(i))

    def test_dynamical_assortativity_valid(self):
        result = eg.dynamical_assortativity(self.hg_uniform)
        self.assertIsInstance(result, float)

    def test_dynamical_assortativity_raises_on_empty(self):
        with self.assertRaises(EasyGraphError):
            eg.dynamical_assortativity(self.hg_empty)

    def test_dynamical_assortativity_raises_on_singleton(self):
        with self.assertRaises(EasyGraphError):
            eg.dynamical_assortativity(self.hg_singleton)

    def test_dynamical_assortativity_raises_on_nonuniform(self):
        with self.assertRaises(EasyGraphError):
            eg.dynamical_assortativity(self.hg_non_uniform)

    def test_degree_assortativity_raises_on_invalid_kind(self):
        with self.assertRaises(EasyGraphError):
            eg.degree_assortativity(self.hg_uniform, kind="invalid")

    def test_degree_assortativity_raises_on_singleton(self):
        with self.assertRaises(EasyGraphError):
            eg.degree_assortativity(self.hg_singleton)

    def test_degree_assortativity_raises_on_empty(self):
        with self.assertRaises(EasyGraphError):
            eg.degree_assortativity(self.hg_empty)


if __name__ == "__main__":
    unittest.main()
