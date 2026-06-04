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


def _communities_to_membership(G, communities):

    if not isinstance(communities, list):
        communities = list(communities)

    N = G.number_of_nodes()
    membership = [-1] * N
    node_index = G.node_index

    for comm_id, community in enumerate(communities):
        for node in community:
            if node in node_index:
                node_id = node_index[node]
                membership[node_id] = comm_id

    return membership


class TestModularity(unittest.TestCase):
    def setUp(self):
        self.G = eg.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        self.DG = eg.DiGraph()
        self.DG.add_edges_from([(0, 1), (1, 2), (2, 0)])

        self.G_weighted = eg.Graph()
        self.G_weighted.add_edge(0, 1, weight=2)
        self.G_weighted.add_edge(1, 2, weight=3)
        self.G_weighted.add_edge(2, 0, weight=1)

        self.G_selfloop = eg.Graph()
        self.G_selfloop.add_edges_from([(0, 0), (1, 1), (0, 1)])

        self.G_empty = eg.Graph()

        self.config = type('Config', (), {
            'is_directed': False
        })()

    def _get_cpp_module(self):
        return _get_cpp_module()

    def test_undirected_modularity(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = [{0, 1}, {2, 3}]
        membership = _communities_to_membership(self.G, communities)
        q = cpeg.cpp_modularity(self.G.cpp(), membership)
        self.assertIsInstance(q, float)
        self.assertGreaterEqual(q, -1.0)
        self.assertLessEqual(q, 1.0)

    def test_directed_modularity(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = [{0, 1, 2}]
        membership = _communities_to_membership(self.DG, communities)
        q = cpeg.cpp_modularity(self.DG.cpp(), membership)
        self.assertIsInstance(q, float)

    def test_weighted_graph(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = [{0, 1}, {2}]
        membership = _communities_to_membership(self.G_weighted, communities)
        q = cpeg.cpp_modularity(self.G_weighted.cpp(), membership, weight="weight")
        self.assertIsInstance(q, float)

    def test_self_loops(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = [{0, 1}]
        membership = _communities_to_membership(self.G_selfloop, communities)
        q = cpeg.cpp_modularity(self.G_selfloop.cpp(), membership)
        self.assertIsInstance(q, float)

    def test_single_community(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = [{0, 1, 2, 3}]
        membership = _communities_to_membership(self.G, communities)
        q = cpeg.cpp_modularity(self.G.cpp(), membership)
        self.assertIsInstance(q, float)

    def test_each_node_its_own_community(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities = [{0}, {1}, {2}, {3}]
        membership = _communities_to_membership(self.G, communities)
        q = cpeg.cpp_modularity(self.G.cpp(), membership)
        self.assertIsInstance(q, float)

    def test_empty_community_list(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        q = cpeg.cpp_modularity(self.G.cpp(), [])
        self.assertEqual(q, 0.0)

    def test_python_cpp_consistency(self):
        cpeg = self._get_cpp_module()
        if cpeg is None:
            self.skipTest("cpp_easygraph module not available")

        communities_py = [{0, 1}, {2, 3}]

        q_py = eg.functions.community.modularity(self.G, communities_py)
        q_cpp = cpeg.cpp_modularity(self.G.cpp(), communities_py)

        self.assertAlmostEqual(q_py, q_cpp, places=5)


if __name__ == "__main__":
    unittest.main()