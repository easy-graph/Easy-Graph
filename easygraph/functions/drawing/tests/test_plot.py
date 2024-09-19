import unittest

import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [
            (1, 4),
            (2, 4),
            ("String", "Bool"),
            (4, 1),
            (0, 4),
            (4, 256),
            ((1, 2), (3, 4)),
        ]
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

    def test_plot_Followers(self):
        eg.functions.plot_Followers(self.ds, self.shs)

    def test_plot_Connected_Communities(self):
        eg.functions.plot_Connected_Communities(self.ds, self.shs)

    def test_plot_Neighborhood_Followers(self):
        eg.functions.plot_Neighborhood_Followers(self.ds, self.shs)

    def test_plot_Betweenness_Centrality(self):
        eg.functions.plot_Betweenness_Centrality(self.ds, self.shs)


if __name__ == "__main__":
    unittest.main()
