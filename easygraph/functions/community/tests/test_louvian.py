import unittest

import easygraph as eg


class TestLouvainCommunityDetection(unittest.TestCase):
    def setUp(self):
        self.graph_simple = eg.Graph()
        self.graph_simple.add_edges_from([(0, 1), (1, 2), (3, 4)])

        self.graph_weighted = eg.Graph()
        self.graph_weighted.add_edges_from(
            [(0, 1, {"weight": 5}), (1, 2, {"weight": 3}), (3, 4, {"weight": 2})]
        )

        self.graph_directed = eg.DiGraph()
        self.graph_directed.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])

        self.graph_disconnected = eg.Graph()
        self.graph_disconnected.add_edges_from([(0, 1), (2, 3), (4, 5)])

        self.graph_single_node = eg.Graph()
        self.graph_single_node.add_node(42)

        self.graph_empty = eg.Graph()

    def test_louvain_communities_simple(self):
        communities = eg.functions.community.louvain_communities(self.graph_simple)
        flat = {node for comm in communities for node in comm}
        self.assertSetEqual(flat, set(self.graph_simple.nodes))

    def test_louvain_communities_weighted(self):
        communities = eg.functions.community.louvain_communities(
            self.graph_weighted, weight="weight"
        )
        flat = {node for comm in communities for node in comm}
        self.assertSetEqual(flat, set(self.graph_weighted.nodes))

    def test_louvain_communities_disconnected(self):
        communities = eg.functions.community.louvain_communities(
            self.graph_disconnected
        )
        flat = {node for comm in communities for node in comm}
        self.assertSetEqual(flat, set(self.graph_disconnected.nodes))

    def test_louvain_communities_single_node(self):
        communities = eg.functions.community.louvain_communities(self.graph_single_node)
        self.assertEqual(len(communities), 1)
        self.assertSetEqual(communities[0], {42})

    def test_louvain_communities_empty_graph(self):
        communities = eg.functions.community.louvain_communities(self.graph_empty)
        self.assertEqual(communities, [])

    def test_louvain_partitions_progressive_size(self):
        partitions = list(eg.functions.community.louvain_partitions(self.graph_simple))
        for partition in partitions:
            total_nodes = sum(len(p) for p in partition)
            self.assertEqual(total_nodes, len(self.graph_simple.nodes))
            flat = [node for part in partition for node in part]
            self.assertEqual(len(flat), len(set(flat)))


if __name__ == "__main__":
    unittest.main()
