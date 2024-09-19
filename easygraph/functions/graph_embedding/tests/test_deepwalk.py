import unittest

import easygraph as eg
import numpy as np


class Test_Deepwalk(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [(1, 4), (2, 4), ("String", "Bool"), (4, 1), (0, 4), (4, 256)]
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

    def test_deepwalk(self):
        for i in self.test_graphs:
            print(eg.deepwalk(i))


if __name__ == "__main__":
    unittest.main()
