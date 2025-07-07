import unittest

import easygraph as eg


class test_classic(unittest.TestCase):
    def setUp(self):
        self.G = eg.datasets.get_graph_karateclub()

    def test_empty_graph(self):
        # print(eg.empty_graph(-1).nodes)
        print(eg.empty_graph(10).nodes)

    def test_path_graph(self):
        eg.path_graph(10, eg.DiGraph)

    def test_complete_graph(self):
        eg.complete_graph(10, eg.DiGraph)

    def test_empty_graph_default(self):
        G = eg.empty_graph()
        self.assertEqual(len(G.nodes), 0)
        self.assertEqual(len(G.edges), 0)

    def test_empty_graph_with_n(self):
        G = eg.empty_graph(5)
        self.assertEqual(set(G.nodes), set(range(5)))
        self.assertEqual(len(G.edges), 0)

    def test_empty_graph_with_custom_nodes(self):
        G = eg.empty_graph(["a", "b", "c"])
        self.assertEqual(set(G.nodes), {"a", "b", "c"})
        self.assertEqual(len(G.edges), 0)

    def test_empty_graph_with_existing_graph(self):
        existing = eg.Graph()
        existing.add_node(999)
        G = eg.empty_graph(3, create_using=existing)
        self.assertIn(0, G.nodes)  # node 0 added
        self.assertEqual(len(G.nodes), 4)  # 999 is retained
        self.assertEqual(len(G.edges), 0)

    def test_path_graph_basic(self):
        G = eg.path_graph(4)
        self.assertEqual(len(G.nodes), 4)
        self.assertEqual(len(G.edges), 3)
        edges = {(u, v) for u, v, _ in G.edges}
        self.assertTrue((0, 1) in edges and (1, 2) in edges and (2, 3) in edges)

    def test_path_graph_with_custom_nodes(self):
        G = eg.path_graph(["x", "y", "z"])
        self.assertEqual(len(G.nodes), 3)
        actual_edges = {(u, v) for u, v, _ in G.edges}
        expected_edges = {("x", "y"), ("y", "z")}
        self.assertEqual(actual_edges, expected_edges)

    def test_complete_graph_basic(self):
        G = eg.complete_graph(4)
        self.assertEqual(len(G.nodes), 4)
        self.assertEqual(len(G.edges), 6)  # n*(n-1)/2 for undirected

    def test_complete_graph_directed(self):
        G = eg.complete_graph(3, create_using=eg.DiGraph())
        self.assertTrue(G.is_directed())
        self.assertEqual(len(G.nodes), 3)
        self.assertEqual(len(G.edges), 6)  # n*(n-1) for directed

    def test_complete_graph_custom_nodes(self):
        G = eg.complete_graph(["a", "b", "c"])
        self.assertEqual(set(G.nodes), {"a", "b", "c"})
        actual_edges = {(u, v) for u, v, _ in G.edges}
        expected_edges = {("a", "b"), ("a", "c"), ("b", "c")}
        self.assertEqual(actual_edges, expected_edges)

    def test_complete_graph_one_node(self):
        G = eg.complete_graph(1)
        self.assertEqual(len(G.nodes), 1)
        self.assertEqual(len(G.edges), 0)

    def test_complete_graph_zero_nodes(self):
        G = eg.complete_graph(0)
        self.assertEqual(len(G.nodes), 0)
        self.assertEqual(len(G.edges), 0)


if __name__ == "__main__":
    unittest.main()
