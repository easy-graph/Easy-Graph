import math
import unittest

from itertools import combinations

import easygraph as eg
import numpy as np


class test_assortativity(unittest.TestCase):
    def setUp(self):
        self.g = eg.get_graph_karateclub()
        self.edges = [(8, 9), (1, 2), (8, 4), (3, 6), (1, 3), (6, 4)]
        self.hg = [
            eg.Hypergraph(num_v=10, e_list=self.edges, e_property=None),
            eg.Hypergraph(num_v=2, e_list=[(0, 1)]),
        ]
        # checked -- num_v cannot be set to negative number

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


if __name__ == "__main__":
    unittest.main()
