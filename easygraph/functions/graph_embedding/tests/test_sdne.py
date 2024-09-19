import unittest

import easygraph as eg
import numpy as np


class Test_Sdne(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [
            (1, 4),
            (2, 4),
            (4, 1),
            (0, 4),
            (4, 256),
            (3.1415926, 0.142857),
            ("bool", "string"),
        ]
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

    def test_parse_args(self):
        print(eg.parse_args())

    def test_get_adj(self):
        print(eg.get_adj(self.test_graphs[-1]))


if __name__ == "__main__":
    unittest.main()
