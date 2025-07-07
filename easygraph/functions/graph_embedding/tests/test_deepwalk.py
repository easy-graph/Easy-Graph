import unittest

import easygraph as eg
import numpy as np


class Test_Deepwalk(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [(1, 4), (2, 4)]
        self.test_graphs = []
        self.test_graphs.append(eg.classes.DiGraph(self.edges))
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

        self.graph = eg.Graph()
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

        self.empty_graph = eg.Graph()

        self.single_node_graph = eg.Graph()
        self.single_node_graph.add_node(0)

    def test_deepwalk(self):
        for i in self.test_graphs:
            print(eg.deepwalk(i))

    def test_deepwalk_output_structure(self):
        emb, sim = eg.deepwalk(
            self.graph,
            dimensions=16,
            walk_length=5,
            num_walks=3,
            window=2,
            min_count=1,
            batch_words=4,
            epochs=5,
        )
        self.assertIsInstance(emb, dict)
        self.assertIsInstance(sim, dict)
        for k, v in emb.items():
            self.assertEqual(len(v), 16)
            self.assertTrue(isinstance(v, np.ndarray))

    def test_deepwalk_similarity_keys_match_nodes(self):
        emb, sim = eg.deepwalk(
            self.graph,
            dimensions=8,
            walk_length=3,
            num_walks=2,
            window=2,
            min_count=1,
            batch_words=2,
            epochs=3,
        )
        self.assertEqual(set(emb.keys()), set(sim.keys()))
        self.assertEqual(set(emb.keys()), set(self.graph.nodes))

    def test_deepwalk_on_single_node(self):
        emb, sim = eg.deepwalk(
            self.single_node_graph,
            dimensions=4,
            walk_length=2,
            num_walks=1,
            window=1,
            min_count=1,
            batch_words=2,
            epochs=2,
        )
        self.assertEqual(len(emb), 1)
        self.assertEqual(list(emb.keys()), [0])
        self.assertEqual(len(emb[0]), 4)

    def test_deepwalk_on_empty_graph(self):
        with self.assertRaises(RuntimeError):
            eg.deepwalk(
                self.empty_graph,
                dimensions=4,
                walk_length=2,
                num_walks=1,
                window=1,
                min_count=1,
                batch_words=2,
                epochs=2,
            )

    def test_deepwalk_walk_length_zero(self):
        emb, sim = eg.deepwalk(
            self.graph,
            dimensions=4,
            walk_length=0,
            num_walks=2,
            window=1,
            min_count=1,
            batch_words=2,
            epochs=2,
        )
        self.assertEqual(len(emb), len(self.graph.nodes))


if __name__ == "__main__":
    unittest.main()
