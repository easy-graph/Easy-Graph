import unittest

import easygraph as eg


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

    def test_simple_graph_radius_1(self):
        ego = eg.functions.community.ego_graph(self.simple_graph, 2, radius=1)
        self.assertSetEqual(set(ego.nodes), {1, 2, 3})

    def test_simple_graph_radius_2(self):
        ego = eg.functions.community.ego_graph(self.simple_graph, 2, radius=2)
        self.assertSetEqual(set(ego.nodes), {0, 1, 2, 3, 4})

    def test_directed_graph(self):
        ego = eg.functions.community.ego_graph(self.directed_graph, 1, radius=1)
        self.assertSetEqual(set(ego.nodes), {1, 2})

    def test_weighted_graph_with_distance(self):
        ego = eg.functions.community.ego_graph(
            self.weighted_graph, 0, radius=2, distance="weight"
        )
        self.assertSetEqual(set(ego.nodes), {0, 1})

    def test_disconnected_graph(self):
        ego = eg.functions.community.ego_graph(self.disconnected_graph, 0, radius=1)
        self.assertSetEqual(set(ego.nodes), {0, 1})

    def test_single_node_graph(self):
        ego = eg.functions.community.ego_graph(self.single_node_graph, 42, radius=1)
        self.assertSetEqual(set(ego.nodes), {42})

    def test_center_false(self):
        ego = eg.functions.community.ego_graph(
            self.simple_graph, 2, radius=1, center=False
        )
        self.assertSetEqual(set(ego.nodes), {1, 3})

    def test_empty_graph(self):
        G = eg.Graph()
        G.add_node("x")
        ego = eg.functions.community.ego_graph(G, "x", radius=1)
        self.assertSetEqual(set(ego.nodes), {"x"})


if __name__ == "__main__":
    unittest.main()
