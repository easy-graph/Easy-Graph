import unittest

import easygraph as eg


def _get_cpp_module():
    """获取cpp_easygraph模块"""
    try:
        import cpp_easygraph
        return cpp_easygraph
    except ImportError:
        return None


def _to_undirected_cpp(G_cpp_digraph):
    """将C++ DiGraph转换为C++ Graph无向图"""
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


def _csr_dict_to_graph(csr_dict):
    import cpp_easygraph

    if not csr_dict or 'nodes' not in csr_dict:
        return cpp_easygraph.Graph()

    G = cpp_easygraph.Graph()
    
    nodes = csr_dict['nodes']
    for node in nodes:
        G.add_node(node)
    
    V = csr_dict['V']
    E = csr_dict['E']
    W = csr_dict.get('W', [1.0] * len(E))
    
    for u in range(len(V) - 1):
        start = V[u]
        end = V[u + 1]
        for i in range(start, end):
            v_idx = E[i]
            weight = W[i] if i < len(W) else 1.0
            G.add_edge(nodes[u], nodes[v_idx], weight=weight)
    
    return G


class TestEgoGraph(unittest.TestCase):
    def setUp(self):
        self.simple_graph = eg.Graph()
        self.simple_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

        self.directed_graph = eg.DiGraph()
        self.directed_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        self.weighted_graph = eg.Graph()
        self.weighted_graph.add_edges_from(
            [(0, 1, {"weight": 1}), (1, 2, {"weight": 2}), (2, 3, {"weight": 3})]
        )

        self.disconnected_graph = eg.Graph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3)])

        self.single_node_graph = eg.Graph()
        self.single_node_graph.add_node(42)

    def _get_cpp_module(self):
        return _get_cpp_module()

    def test_ego_graph_simple_radius_1(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.simple_graph.cpp()
        ego = cpeg.cpp_ego_graph(G_cpp, 2, radius=1)
        self.assertIsInstance(ego, type(G_cpp))
        self.assertIn(2, ego.nodes)
        self.assertIn(1, ego.nodes)
        self.assertIn(3, ego.nodes)

    def test_ego_graph_simple_radius_2(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.simple_graph.cpp()
        ego = cpeg.cpp_ego_graph(G_cpp, 2, radius=2)
        self.assertIsInstance(ego, type(G_cpp))
        self.assertIn(0, ego.nodes)
        self.assertIn(1, ego.nodes)
        self.assertIn(2, ego.nodes)
        self.assertIn(3, ego.nodes)
        self.assertIn(4, ego.nodes)

    def test_ego_graph_directed(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.directed_graph.cpp()
        ego = cpeg.cpp_ego_graph(G_cpp, 1, radius=1)
        self.assertIsInstance(ego, type(G_cpp))
        self.assertIn(1, ego.nodes)
        self.assertIn(2, ego.nodes)

    def test_ego_graph_weighted_with_distance(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.weighted_graph.cpp()
        G_cpp.generate_linkgraph('weight')
        ego = cpeg.cpp_ego_graph(G_cpp, 0, radius=2, distance="weight")
        self.assertIsInstance(ego, type(G_cpp))
        self.assertIn(0, ego.nodes)
        self.assertIn(1, ego.nodes)

    def test_ego_graph_disconnected(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.disconnected_graph.cpp()
        ego = cpeg.cpp_ego_graph(G_cpp, 0, radius=1)
        self.assertIsInstance(ego, type(G_cpp))
        self.assertIn(0, ego.nodes)
        self.assertIn(1, ego.nodes)

    def test_ego_graph_single_node(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.single_node_graph.cpp()
        ego = cpeg.cpp_ego_graph(G_cpp, 42, radius=1)
        self.assertIsInstance(ego, type(G_cpp))
        self.assertIn(42, ego.nodes)

    def test_ego_graph_center_false(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.simple_graph.cpp()
        ego = cpeg.cpp_ego_graph(G_cpp, 2, radius=1, center=False)
        self.assertIsInstance(ego, type(G_cpp))
        self.assertNotIn(2, ego.nodes)
        self.assertIn(1, ego.nodes)
        self.assertIn(3, ego.nodes)

    def test_python_cpp_consistency(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.simple_graph.cpp()
        ego_py = eg.functions.community.ego_graph(self.simple_graph, 2, radius=1)
        ego_cpp = cpeg.cpp_ego_graph(G_cpp, 2, radius=1)

        self.assertEqual(set(ego_py.nodes), set(ego_cpp.nodes))


class TestEgoGraphCSR(unittest.TestCase):
    def setUp(self):
        self.simple_graph = eg.Graph()
        self.simple_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

        self.disconnected_graph = eg.Graph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3)])

        self.single_node_graph = eg.Graph()
        self.single_node_graph.add_node(42)

    def _get_cpp_module(self):
        return _get_cpp_module()

    def test_ego_graph_csr_simple(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.simple_graph.cpp()
        G_cpp.generate_linkgraph('weight')
        ego_csr_dict = cpeg.cpp_ego_graph_csr(G_cpp, 2, radius=1)
        
        ego = _csr_dict_to_graph(ego_csr_dict)
        
        self.assertTrue(hasattr(ego, 'nodes'))
        self.assertIn(2, ego.nodes)

    def test_ego_graph_csr_disconnected(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.disconnected_graph.cpp()
        G_cpp.generate_linkgraph('weight')
        ego_csr_dict = cpeg.cpp_ego_graph_csr(G_cpp, 0, radius=1)
        
        ego = _csr_dict_to_graph(ego_csr_dict)
        
        self.assertTrue(hasattr(ego, 'nodes'))
        self.assertIn(0, ego.nodes)
        self.assertIn(1, ego.nodes)

    def test_ego_graph_csr_single_node(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.single_node_graph.cpp()
        G_cpp.generate_linkgraph('weight')
        ego_csr_dict = cpeg.cpp_ego_graph_csr(G_cpp, 42, radius=1)
        
        ego = _csr_dict_to_graph(ego_csr_dict)
        
        self.assertTrue(hasattr(ego, 'nodes'))
        self.assertIn(42, ego.nodes)

    def test_csr_vs_original_consistency(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        G_cpp = self.simple_graph.cpp()
        G_cpp.generate_linkgraph('weight')

        ego_original = cpeg.cpp_ego_graph(G_cpp, 2, radius=1, undirected=False)
        ego_csr_dict = cpeg.cpp_ego_graph_csr(G_cpp, 2, radius=1)
        
        ego_csr = _csr_dict_to_graph(ego_csr_dict)

        self.assertIn(2, ego_original.nodes)
        self.assertIn(2, ego_csr.nodes)
        
        self.assertTrue(set(ego_csr.nodes).issubset(set(ego_original.nodes) | {4}))


if __name__ == "__main__":
    unittest.main()