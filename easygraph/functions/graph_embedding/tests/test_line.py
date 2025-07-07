import unittest

import easygraph as eg
import numpy as np


class Test_LINE(unittest.TestCase):
    def setUp(self):
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.graph = eg.Graph()
        self.graph.add_edges_from(self.edges)

    def test_output_is_dict_with_correct_dim(self):
        model = eg.functions.graph_embedding.LINE(
            dimension=16, walk_length=10, walk_num=5, order=1
        )
        emb = model(self.graph, return_dict=True)
        self.assertIsInstance(emb, dict)
        for v in emb.values():
            self.assertEqual(len(v), 16)

    def test_output_as_matrix(self):
        model = eg.functions.graph_embedding.LINE(
            dimension=8, walk_length=5, walk_num=3, order=1
        )
        emb = model(self.graph, return_dict=False)
        self.assertEqual(emb.shape, (len(self.graph.nodes), 8))

    def test_output_with_order_2(self):
        model = eg.functions.graph_embedding.LINE(
            dimension=16, walk_length=10, walk_num=5, order=2
        )
        emb = model(self.graph)
        for vec in emb.values():
            self.assertEqual(len(vec), 16)

    def test_output_with_order_3_combination(self):
        model = eg.functions.graph_embedding.LINE(
            dimension=16, walk_length=10, walk_num=5, order=3
        )
        emb = model(self.graph)
        for vec in emb.values():
            self.assertEqual(len(vec), 16)

    def test_directed_graph(self):
        g = eg.DiGraph()
        g.add_edges_from(self.edges)
        model = eg.functions.graph_embedding.LINE(
            dimension=8, walk_length=5, walk_num=3, order=1
        )
        emb = model(g)
        self.assertEqual(len(emb), len(g.nodes))

    def test_empty_graph_raises(self):
        g = eg.Graph()
        model = eg.functions.graph_embedding.LINE(
            dimension=8, walk_length=5, walk_num=3, order=1
        )
        with self.assertRaises(Exception):
            _ = model(g)

    def test_embeddings_are_normalized(self):
        model = eg.functions.graph_embedding.LINE(
            dimension=16, walk_length=10, walk_num=5, order=1
        )
        emb = model(self.graph)
        for vec in emb.values():
            norm = np.linalg.norm(vec)
            self.assertTrue(np.isclose(norm, 1.0, atol=1e-5))

    def test_embedding_value_finiteness(self):
        model = eg.functions.graph_embedding.LINE(
            dimension=16, walk_length=10, walk_num=5, order=1
        )
        emb = model(self.graph)
        for vec in emb.values():
            self.assertTrue(np.all(np.isfinite(vec)))
