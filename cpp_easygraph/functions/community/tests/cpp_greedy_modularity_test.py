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


class TestGreedyModularity(unittest.TestCase):
    def setUp(self):
        self.G_simple = eg.Graph()
        self.G_simple.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        self.G_disconnected = eg.Graph()
        self.G_disconnected.add_edges_from([(0, 1), (2, 3), (4, 5)])

        self.G_weighted = eg.Graph()
        self.G_weighted.add_edge(0, 1, weight=5)
        self.G_weighted.add_edge(1, 2, weight=3)
        self.G_weighted.add_edge(2, 0, weight=2)
        self.G_weighted.add_edge(3, 4, weight=1)

        self.G_single = eg.Graph()
        self.G_single.add_node(42)

        self.G_empty = eg.Graph()

    def _get_cpp_module(self):
        return _get_cpp_module()

    def test_greedy_simple(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = cpeg.cpp_greedy_modularity_communities(self.G_simple.cpp())
        self.assertIsInstance(communities, list)
        flat = {node for comm in communities for node in comm}
        self.assertSetEqual(flat, set(self.G_simple.nodes))

    def test_greedy_weighted(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = cpeg.cpp_greedy_modularity_communities(self.G_weighted.cpp(), weight="weight")
        self.assertIsInstance(communities, list)
        flat = {node for comm in communities for node in comm}
        self.assertSetEqual(flat, set(self.G_weighted.nodes))

    def test_greedy_disconnected(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = cpeg.cpp_greedy_modularity_communities(self.G_disconnected.cpp())
        self.assertIsInstance(communities, list)
        flat = {node for comm in communities for node in comm}
        self.assertSetEqual(flat, set(self.G_disconnected.nodes))

    def test_greedy_single_node(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = cpeg.cpp_greedy_modularity_communities(self.G_single.cpp())
        self.assertEqual(len(communities), 1)
        self.assertIn(42, communities[0])

    def test_greedy_empty_graph(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = cpeg.cpp_greedy_modularity_communities(self.G_empty.cpp())
        self.assertEqual(communities, [])

    def test_python_cpp_consistency(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities_cpp = cpeg.cpp_greedy_modularity_communities(self.G_simple.cpp())
        communities_py = eg.functions.community.greedy_modularity_communities(self.G_simple)

        self.assertEqual(len(communities_cpp), len(communities_py))

        flat_cpp = {node for comm in communities_cpp for node in comm}
        flat_py = {node for comm in communities_py for node in comm}
        self.assertEqual(flat_cpp, flat_py)


if __name__ == "__main__":
    unittest.main()