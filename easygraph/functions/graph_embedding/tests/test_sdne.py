import unittest

import easygraph as eg


class Test_Sdne(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [
            (1, 4),
            (2, 4),
            (4, 1),
            (0, 4),
            (4, 3),
        ]
        self.test_graphs = []
        self.test_graphs.append(eg.classes.DiGraph(self.edges))
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

    def test_sdne(self):
        sdne = eg.SDNE(
            graph=self.test_graphs[0],
            node_size=len(self.test_graphs[0].nodes),
            nhid0=128,
            nhid1=64,
            dropout=0.025,
            alpha=2e-2,
            beta=10,
        )
        # todo add test
        # emb = sdne.train(sdne)
