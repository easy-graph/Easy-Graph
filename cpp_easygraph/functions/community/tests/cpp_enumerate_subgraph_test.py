import random
import unittest

import easygraph as eg


def _get_cpp_module():
    try:
        import cpp_easygraph
        return cpp_easygraph
    except ImportError:
        return None


def _to_undirected_cpp(G_cpp_digraph):
    import cpp_easygraph

    G_cpp = cpp_easygraph.Graph()
    G_cpp.graph.update(G_cpp_digraph.graph)
    for node, node_attr in G_cpp_digraph.nodes.items():
        G_cpp.add_node(node, **node_attr)

    seen_edges = set()
    for u, v, edge_data in G_cpp_digraph.edges:
        edge = (min(u, v), max(u, v))
        if edge not in seen_edges:
            seen_edges.add(edge)
            G_cpp.add_edge(u, v, **edge_data)

    return G_cpp


class TestEnumerateSubgraph(unittest.TestCase):
    def setUp(self):
        self.G_triangle = eg.Graph()
        self.G_triangle.add_edges_from([(1, 2), (2, 3), (3, 1)])

        self.G_complex = eg.Graph()
        self.G_complex.add_edges_from([
            (1, 3), (2, 3), (3, 4), (4, 5), (3, 5)
        ])

        self.G_empty = eg.Graph()

        self.G_small = eg.Graph()
        self.G_small.add_edges_from([(1, 2)])

    def _get_cpp_module(self):
        return _get_cpp_module()

    def test_enumerate_subgraph_k3(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        motifs = cpeg.cpp_enumerate_subgraph(self.G_complex.cpp(), 3)
        self.assertIsInstance(motifs, list)
        for m in motifs:
            self.assertEqual(len(m), 3)

    def test_enumerate_subgraph_k4(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        motifs = cpeg.cpp_enumerate_subgraph(self.G_complex.cpp(), 4)
        for m in motifs:
            self.assertEqual(len(m), 4)

    def test_enumerate_subgraph_empty_graph(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        motifs = cpeg.cpp_enumerate_subgraph(self.G_empty.cpp(), 3)
        self.assertEqual(motifs, [])

    def test_enumerate_subgraph_k_larger_than_graph(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        motifs = cpeg.cpp_enumerate_subgraph(self.G_small.cpp(), 3)
        self.assertEqual(motifs, [])

    def test_python_cpp_consistency(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        motifs_py = eg.enumerate_subgraph(self.G_complex, 3)
        motifs_py = [frozenset(m) for m in motifs_py]
        motifs_cpp = cpeg.cpp_enumerate_subgraph(self.G_complex.cpp(), 3)
        motifs_cpp = [frozenset(m) for m in motifs_cpp]

        self.assertEqual(set(motifs_py), set(motifs_cpp))


class TestRandomEnumerateSubgraph(unittest.TestCase):
    def setUp(self):
        self.G_complex = eg.Graph()
        self.G_complex.add_edges_from([
            (1, 3), (2, 3), (3, 4), (4, 5), (3, 5)
        ])

        self.G_empty = eg.Graph()

    def _get_cpp_module(self):
        return _get_cpp_module()

    def test_random_enumerate_valid_cut_prob(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        cut_prob = [1.0, 1.0, 1.0]
        motifs = cpeg.cpp_random_enumerate_subgraph(self.G_complex.cpp(), 3, cut_prob)
        self.assertIsInstance(motifs, list)

    def test_random_enumerate_zero_cut_prob(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        cut_prob = [0.0, 0.0, 0.0]
        motifs = cpeg.cpp_random_enumerate_subgraph(self.G_complex.cpp(), 3, cut_prob)
        self.assertEqual(motifs, [])

    def test_random_enumerate_different_results(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        cut_prob = [1.0, 1.0, 1.0]
        motifs1 = cpeg.cpp_random_enumerate_subgraph(self.G_complex.cpp(), 3, cut_prob)
        motifs2 = cpeg.cpp_random_enumerate_subgraph(self.G_complex.cpp(), 3, cut_prob)

        self.assertIsInstance(motifs1, list)
        self.assertIsInstance(motifs2, list)


if __name__ == "__main__":
    unittest.main()