import unittest

import easygraph as eg


class TestEasyGraph(unittest.TestCase):
    def setUp(self):
        self.G = eg.Graph()

    def test_add_single_node(self):
        self.G.add_node(1)
        self.assertIn(1, self.G.nodes)

    def test_add_multiple_nodes(self):
        self.G.add_nodes([2, 3, 4])
        for node in [2, 3, 4]:
            self.assertIn(node, self.G.nodes)

    def test_add_node_with_attributes(self):
        self.G.add_node("node", color="red")
        self.assertEqual(self.G.nodes["node"]["color"], "red")

    def test_add_single_edge(self):
        self.G.add_edge(1, 2)
        self.assertTrue(self.G.has_edge(1, 2))
        self.assertTrue(self.G.has_edge(2, 1))

    def test_add_edge_with_weight(self):
        self.G.add_edge("a", "b", weight=10)
        self.assertEqual(self.G["a"]["b"]["weight"], 10)

    def test_add_edges(self):
        self.G.add_edges([(1, 2), (2, 3)], edges_attr=[{"weight": 5}, {"weight": 6}])
        self.assertEqual(self.G[1][2]["weight"], 5)
        self.assertEqual(self.G[2][3]["weight"], 6)

    def test_remove_node(self):
        self.G.add_node(10)
        self.G.remove_node(10)
        self.assertNotIn(10, self.G.nodes)

    def test_remove_edge(self):
        self.G.add_edge(1, 2)
        self.G.remove_edge(1, 2)
        self.assertFalse(self.G.has_edge(1, 2))

    def test_neighbors(self):
        self.G.add_edges([(1, 2), (1, 3)])
        neighbors = list(self.G.neighbors(1))
        self.assertIn(2, neighbors)
        self.assertIn(3, neighbors)

    def test_subgraph(self):
        self.G.add_edges([(1, 2), (2, 3), (3, 4)])
        subG = self.G.nodes_subgraph([2, 3])
        self.assertIn(2, subG.nodes)
        self.assertIn(3, subG.nodes)
        self.assertTrue(subG.has_edge(2, 3))
        self.assertFalse(subG.has_edge(3, 4))

    def test_ego_subgraph(self):
        self.G.add_edges([(1, 2), (2, 3), (2, 4)])
        ego = self.G.ego_subgraph(2)
        self.assertIn(2, ego.nodes)
        self.assertIn(1, ego.nodes)
        self.assertIn(3, ego.nodes)
        self.assertIn(4, ego.nodes)

    def test_to_index_node_graph(self):
        self.G.add_edges([("a", "b"), ("b", "c")])
        G_index, index_of_node, node_of_index = self.G.to_index_node_graph()
        self.assertEqual(len(G_index.nodes), 3)
        self.assertTrue(all(isinstance(k, int) for k in G_index.nodes))

    def test_directed_conversion(self):
        self.G.add_edge(1, 2)
        H = self.G.to_directed()
        self.assertTrue(H.is_directed())
        self.assertTrue(H.has_edge(1, 2))
        self.assertTrue(H.has_edge(2, 1))

    def test_clone_graph(self):
        self.G.add_edges([(1, 2), (2, 3)])
        G_clone = self.G.copy()
        self.assertTrue(G_clone.has_edge(1, 2))
        self.assertTrue(G_clone.has_edge(2, 3))

    def test_degree(self):
        self.G.add_edge(1, 2, weight=5)
        deg = self.G.degree()
        self.assertEqual(deg[1], 5)
        self.assertEqual(deg[2], 5)

    def test_size(self):
        self.G.add_edges([(1, 2), (2, 3)])
        self.assertEqual(self.G.size(), 2)

    def test_edge_weight_default(self):
        self.G.add_edge(4, 5)
        self.assertEqual(self.G[4][5].get("weight", 1), 1)

    def test_node_index_mappings(self):
        self.G.add_nodes([10, 20, 30])
        index2node = self.G.index2node
        node_index = self.G.node_index
        for i, node in index2node.items():
            self.assertEqual(node_index[node], i)

    def test_graph_order(self):
        self.G.add_nodes([1, 2, 3])
        self.assertEqual(self.G.order(), 3)

    def test_graph_size_with_weight(self):
        self.G.add_edges([(1, 2), (2, 3)], edges_attr=[{"weight": 4}, {"weight": 6}])
        self.assertEqual(self.G.size(weight="weight"), 10.0)

    def test_clear_cache(self):
        self.G.add_edge(1, 2)
        _ = self.G.edges
        self.assertIn("edge", self.G.cache)
        self.G._clear_cache()
        self.assertEqual(len(self.G.cache), 0)
