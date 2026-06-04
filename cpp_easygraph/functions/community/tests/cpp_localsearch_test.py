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


class TestLocalsearch(unittest.TestCase):
    def setUp(self):
        self.G_simple = eg.Graph()
        self.G_simple.add_edges_from([(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5)])

        self.G_disconnected = eg.Graph()
        self.G_disconnected.add_edges_from([(0, 1), (2, 3), (4, 5)])

        self.G_single = eg.Graph()
        self.G_single.add_node(42)

        self.G_empty = eg.Graph()

    def _get_cpp_module(self):
        return _get_cpp_module()

    def test_localsearch_simple(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_simple.cpp()
        G_cpp.generate_linkgraph('weight')
        result = cpeg.cpp_localsearch(G_cpp, seed=163)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        y_partition = result[3]
        self.assertIsInstance(y_partition, dict)

        all_nodes = set(self.G_simple.nodes)
        result_nodes = set(y_partition.keys())
        self.assertEqual(all_nodes, result_nodes)

    def test_localsearch_disconnected(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_disconnected.cpp()
        G_cpp.generate_linkgraph('weight')
        result = cpeg.cpp_localsearch(G_cpp, seed=163)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        y_partition = result[3]
        self.assertIsInstance(y_partition, dict)

        all_nodes = set(self.G_disconnected.nodes)
        result_nodes = set(y_partition.keys())
        self.assertEqual(all_nodes, result_nodes)

    def test_localsearch_single_node(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_single.cpp()
        G_cpp.generate_linkgraph('weight')
        result = cpeg.cpp_localsearch(G_cpp, seed=163)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        y_partition = result[3]
        self.assertIsInstance(y_partition, dict)
        self.assertIn(42, y_partition)

    def test_localsearch_empty_graph(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_empty.cpp()
        G_cpp.generate_linkgraph('weight')
        result = cpeg.cpp_localsearch(G_cpp, seed=163)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        y_partition = result[3]
        self.assertIsInstance(y_partition, dict)

    def test_localsearch_maximum_tree_true(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_simple.cpp()
        G_cpp.generate_linkgraph('weight')
        result = cpeg.cpp_localsearch(G_cpp, maximum_tree=True, seed=163)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        y_partition = result[3]
        self.assertIsInstance(y_partition, dict)

    def test_localsearch_maximum_tree_false(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_simple.cpp()
        G_cpp.generate_linkgraph('weight')
        result = cpeg.cpp_localsearch(G_cpp, maximum_tree=False, seed=163)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        y_partition = result[3]
        self.assertIsInstance(y_partition, dict)

    def test_localsearch_auto_choose_centers(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_simple.cpp()
        G_cpp.generate_linkgraph('weight')
        result = cpeg.cpp_localsearch(G_cpp, auto_choose_centers=True, seed=163)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        y_partition = result[3]
        self.assertIsInstance(y_partition, dict)

    def test_localsearch_center_num(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_simple.cpp()
        G_cpp.generate_linkgraph('weight')
        result = cpeg.cpp_localsearch(G_cpp, center_num=2, seed=163)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        y_partition = result[3]
        self.assertIsInstance(y_partition, dict)

    def test_localsearch_with_seed_reproducibility(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.G_simple.cpp()
        G_cpp.generate_linkgraph('weight')

        result1 = cpeg.cpp_localsearch(G_cpp, seed=42)
        result2 = cpeg.cpp_localsearch(G_cpp, seed=42)

        self.assertIsInstance(result1, tuple)
        self.assertIsInstance(result2, tuple)
        y_partition1 = result1[3]
        y_partition2 = result2[3]
        self.assertEqual(y_partition1.keys(), y_partition2.keys())


if __name__ == "__main__":
    unittest.main()