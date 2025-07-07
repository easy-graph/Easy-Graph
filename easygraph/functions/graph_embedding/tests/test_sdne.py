import unittest

import easygraph as eg
import numpy as np
import torch


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
        self.graph = eg.DiGraph()
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def test_sdne_model_instantiation(self):
        model = eg.SDNE(
            graph=self.graph,
            node_size=len(self.graph.nodes),
            nhid0=32,
            nhid1=16,
            dropout=0.05,
            alpha=0.01,
            beta=5.0,
        )
        self.assertIsInstance(model, eg.SDNE)

    def test_sdne_training_embedding_output(self):
        model = eg.SDNE(
            graph=self.graph,
            node_size=len(self.graph.nodes),
            nhid0=16,
            nhid1=8,
            dropout=0.05,
            alpha=0.01,
            beta=5.0,
        )
        embedding = model.train(
            model=model,
            epochs=5,
            lr=0.01,
            bs=2,
            step_size=2,
            gamma=0.9,
            nu1=1e-5,
            nu2=1e-4,
            device=self.device,
            output="test.emb",
        )
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (len(self.graph.nodes), 8))

    def test_savector_output_shape(self):
        adj, _ = eg.get_adj(self.graph)
        model = eg.SDNE(
            graph=self.graph,
            node_size=len(self.graph.nodes),
            nhid0=16,
            nhid1=8,
            dropout=0.05,
            alpha=0.01,
            beta=5.0,
        )
        with torch.no_grad():
            emb = model.savector(adj)
        self.assertEqual(emb.shape, (len(self.graph.nodes), 8))

    def test_get_adj_shape_and_symmetry(self):
        adj, node_count = eg.get_adj(self.graph)
        self.assertEqual(adj.shape[0], node_count)
        self.assertTrue(torch.equal(adj, adj.T))  # check symmetry for undirected

    def test_training_on_empty_graph(self):
        empty_graph = eg.Graph()
        model = eg.SDNE(
            graph=empty_graph,
            node_size=0,
            nhid0=8,
            nhid1=4,
            dropout=0.05,
            alpha=0.01,
            beta=5.0,
        )
        with self.assertRaises(ValueError):
            model.train(model=model, epochs=5, device=self.device)
