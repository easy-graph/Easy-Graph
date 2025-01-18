import unittest

import easygraph as eg
import numpy as np

from easygraph.functions.graph_embedding.NOBE import NOBE
from easygraph.functions.graph_embedding.NOBE import NOBE_GA


class Test_Nobe(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [(1, 4), (2, 4), (4, 1), (0, 4)]
        self.test_graphs = [eg.classes.DiGraph(self.edges)]
        self.test_undirected_graphs = [eg.classes.Graph(self.edges)]
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

    #
    def test_NOBE(self):
        for i in self.test_graphs:
            NOBE(i, K=1)

    def test_NOBE_GA(self):
        for i in self.test_undirected_graphs:
            NOBE_GA(i, K=1)


# if __name__ == "__main__":
#     unittest.main()
