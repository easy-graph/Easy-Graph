import random
import unittest

import easygraph as eg


class TestMotif:
    @classmethod
    def setup_class(self):
        self.G = eg.Graph()
        self.G.add_nodes_from([1, 2, 3, 4, 5])
        self.G.add_edges_from([(1, 3), (2, 3), (3, 4), (4, 5), (3, 5)])

    def test_esu(self):
        res = eg.enumerate_subgraph(self.G, 3)
        res = [list(x) for x in res]
        exp_res = [{1, 3, 4}, {1, 2, 3}, {1, 3, 5}, {2, 3, 5}, {2, 3, 4}, {3, 4, 5}]
        exp_res = [list(x) for x in exp_res]
        assert sorted(res) == sorted(exp_res)


class TestMotifEnumeration(unittest.TestCase):
    def setUp(self):
        # Triangle plus a tail
        self.G = eg.Graph()
        self.G.add_edges_from(
            [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5)]  # triangle  # tail
        )

    def test_esu_enumeration_correct(self):
        motifs = eg.enumerate_subgraph(self.G, 3)
        motifs = [frozenset(m) for m in motifs]
        expected = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}]
        expected = [frozenset(x) for x in expected]
        self.assertTrue(all(m in motifs for m in expected))
        for m in motifs:
            self.assertEqual(len(m), 3)
            self.assertTrue(isinstance(m, frozenset))

    def test_empty_graph(self):
        G = eg.Graph()
        motifs = eg.enumerate_subgraph(G, 3)
        self.assertEqual(motifs, [])

    def test_graph_smaller_than_k(self):
        G = eg.Graph()
        G.add_edges_from([(1, 2)])
        motifs = eg.enumerate_subgraph(G, 3)
        self.assertEqual(motifs, [])

    def test_k_equals_1(self):
        G = eg.Graph()
        G.add_nodes_from([1, 2, 3])
        motifs = eg.enumerate_subgraph(G, 1)
        expected = [{1}, {2}, {3}]
        motifs = [set(m) for m in motifs]
        self.assertEqual(sorted(motifs), sorted(expected))

    def test_random_enumerate_cut_prob_valid(self):
        random.seed(0)
        cut_prob = [1.0] * 3
        motifs = eg.random_enumerate_subgraph(self.G, 3, cut_prob)
        for m in motifs:
            self.assertEqual(len(m), 3)

    def test_random_enumerate_cut_prob_invalid_length(self):
        cut_prob = [1.0, 0.9]
        with self.assertRaises(eg.EasyGraphError):
            eg.random_enumerate_subgraph(self.G, 3, cut_prob)

    def test_random_enumerate_zero_cut_prob(self):
        cut_prob = [0.0, 0.0, 0.0]
        motifs = eg.random_enumerate_subgraph(self.G, 3, cut_prob)
        self.assertEqual(motifs, [])

    def test_directed_graph_enumeration(self):
        DG = eg.DiGraph()
        DG.add_edges_from([(1, 2), (2, 3), (3, 1)])
        motifs = eg.enumerate_subgraph(DG, 3)
        motifs = [set(m) for m in motifs]
        self.assertIn({1, 2, 3}, motifs)

    def test_multigraph_error(self):
        MG = eg.MultiGraph()
        MG.add_edges_from([(1, 2), (2, 3)])
        with self.assertRaises(eg.EasyGraphNotImplemented):
            eg.enumerate_subgraph(MG, 3)
        with self.assertRaises(eg.EasyGraphNotImplemented):
            eg.random_enumerate_subgraph(MG, 3, [1.0] * 3)
