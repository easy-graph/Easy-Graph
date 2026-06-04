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


class TestLPA(unittest.TestCase):
    def setUp(self):
        self.G_simple = eg.Graph()
        self.G_simple.add_edges_from([(0, 1), (1, 2), (3, 4)])

        self.G_weighted = eg.Graph()
        self.G_weighted.add_edges_from([
            (0, 1, {"weight": 3}),
            (1, 2, {"weight": 2}),
            (2, 0, {"weight": 4}),
            (3, 4, {"weight": 1}),
        ])

        self.G_disconnected = eg.Graph()
        self.G_disconnected.add_edges_from([(0, 1), (2, 3), (4, 5)])

        self.G_single = eg.Graph()
        self.G_single.add_node(42)

        self.G_empty = eg.Graph()

    def _get_cpp_module(self):
        return _get_cpp_module()

    def test_lpa_simple(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        result = cpeg.cpp_LPA(self.G_simple.cpp())
        self.assertIsInstance(result, dict)

        all_nodes = set()
        for community in result.values():
            all_nodes.update(community)
        self.assertEqual(all_nodes, set(self.G_simple.nodes))

    def test_lpa_weighted(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        result = cpeg.cpp_LPA(self.G_weighted.cpp())
        self.assertIsInstance(result, dict)

        all_nodes = set(self.G_weighted.nodes)
        result_nodes = set()
        for community in result.values():
            result_nodes.update(community)
        self.assertEqual(all_nodes, result_nodes)

    def test_lpa_disconnected(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        result = cpeg.cpp_LPA(self.G_disconnected.cpp())
        self.assertIsInstance(result, dict)

        all_nodes = set(self.G_disconnected.nodes)
        result_nodes = set()
        for community in result.values():
            result_nodes.update(community)
        self.assertEqual(all_nodes, result_nodes)

    def test_lpa_single_node(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        result = cpeg.cpp_LPA(self.G_single.cpp())
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        for community in result.values():
            self.assertIn(42, community)

    def test_lpa_empty_graph(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        result = cpeg.cpp_LPA(self.G_empty.cpp())
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})

    def test_python_cpp_consistency(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        result_cpp = cpeg.cpp_LPA(self.G_simple.cpp())
        result_py = eg.functions.community.LPA(self.G_simple)

        self.assertEqual(len(result_cpp), len(result_py))

        cpp_nodes = set()
        for community in result_cpp.values():
            cpp_nodes.update(community)
        
        py_nodes = set()
        for community in result_py.values():
            py_nodes.update(community)
        
        self.assertEqual(cpp_nodes, py_nodes)


if __name__ == "__main__":
    unittest.main()