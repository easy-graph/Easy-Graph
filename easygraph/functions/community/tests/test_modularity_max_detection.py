import unittest

import easygraph as eg


class TestGreedyModularityCommunities(unittest.TestCase):
    def setUp(self):
        # A simple connected graph
        self.graph_simple = eg.Graph()
        self.graph_simple.add_edges_from([(0, 1), (1, 2), (3, 4)])

        # A weighted graph
        self.graph_weighted = eg.Graph()
        self.graph_weighted.add_edges_from(
            [(0, 1, {"weight": 3}), (1, 2, {"weight": 2}), (3, 4, {"weight": 1})]
        )

        # A fully connected graph (clique)
        self.graph_clique = eg.Graph()
        self.graph_clique.add_edges_from([(0, 1), (0, 2), (1, 2)])

        # A disconnected graph
        self.graph_disconnected = eg.Graph()
        self.graph_disconnected.add_edges_from([(0, 1), (2, 3), (4, 5)])

        # A graph with a single node
        self.graph_single_node = eg.Graph()
        self.graph_single_node.add_node(42)

        # An empty graph
        self.graph_empty = eg.Graph()

    def test_communities_simple(self):
        result = eg.functions.community.greedy_modularity_communities(self.graph_simple)
        flat_nodes = {node for group in result for node in group}
        self.assertSetEqual(flat_nodes, set(self.graph_simple.nodes))

    def test_communities_weighted(self):
        result = eg.functions.community.greedy_modularity_communities(
            self.graph_weighted
        )
        flat_nodes = {node for group in result for node in group}
        self.assertSetEqual(flat_nodes, set(self.graph_weighted.nodes))

    def test_communities_clique(self):
        result = eg.functions.community.greedy_modularity_communities(self.graph_clique)
        self.assertEqual(len(result), 1)
        self.assertSetEqual(result[0], set(self.graph_clique.nodes))

    def test_communities_disconnected(self):
        result = eg.functions.community.greedy_modularity_communities(
            self.graph_disconnected
        )
        flat_nodes = {node for group in result for node in group}
        self.assertSetEqual(flat_nodes, set(self.graph_disconnected.nodes))

    def test_communities_single_node(self):
        with self.assertRaises(SystemExit):
            eg.functions.community.greedy_modularity_communities(self.graph_single_node)

    def test_communities_empty_graph(self):
        with self.assertRaises(SystemExit):
            eg.functions.community.greedy_modularity_communities(self.graph_empty)

    def test_correct_partition_disjoint(self):
        result = eg.functions.community.greedy_modularity_communities(
            self.graph_disconnected
        )
        all_nodes = [node for group in result for node in group]
        self.assertEqual(len(all_nodes), len(set(all_nodes)))

    def test_communities_sorted_by_size(self):
        result = eg.functions.community.greedy_modularity_communities(
            self.graph_disconnected
        )
        sizes = [len(group) for group in result]
        self.assertEqual(sizes, sorted(sizes, reverse=True))


if __name__ == "__main__":
    unittest.main()
