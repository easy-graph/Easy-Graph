import math
import unittest

import easygraph as eg


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


if __name__ == "__main__":
    unittest.main()
