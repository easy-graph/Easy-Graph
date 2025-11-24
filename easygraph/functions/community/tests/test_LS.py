import unittest

import easygraph as eg

from easygraph import LS_degree_communities

class TestLSDetection(unittest.TestCase):
    def setUp(self):
        self.graph_simple = eg.Graph()
        self.graph_simple.add_edges_from([(0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5)])

        self.graph_disconnected = eg.Graph()
        self.graph_disconnected.add_edges_from([(0, 1), (2, 3), (4, 5)])

        self.graph_single_node = eg.Graph()
        self.graph_single_node.add_node(42)

        self.graph_empty = eg.Graph()

    def test_LS_simple(self):
        _, _, _, _, communities, _ = LS_degree_communities(self.graph_simple, maximum_tree=True, isdraw = False, seed=163)
        flat = set().union(*communities.values())
        self.assertSetEqual(flat, set(self.graph_simple.nodes))

    def test_LS_disconnected(self):
        _, _, _, _, communities, _ = LS_degree_communities(self.graph_disconnected, maximum_tree=True, isdraw = False, seed=163)
        flat =set().union(*communities.values())
        self.assertSetEqual(flat, set(self.graph_disconnected.nodes))

    def test_LS_single_node(self):
        _, _, _, _, communities, _ = LS_degree_communities(self.graph_single_node, maximum_tree=True, isdraw = False, seed=163)
        flat = set().union(communities.keys())
        self.assertEqual(len(flat), 1)
        self.assertSetEqual(flat, {42})

    def test_LS_empty_graph(self):
        _, _, _, _, communities, _ = LS_degree_communities(self.graph_empty, maximum_tree=True, isdraw = False, seed=163)
        self.assertEqual(communities, {})

    def test_LS_partitions_progressive_size(self):
        _, _, _, _, communities, _ = LS_degree_communities(self.graph_simple, maximum_tree=True, isdraw = False, seed=163)
        total_nodes = sum(len(members) for center, members in communities.items())
        self.assertEqual(total_nodes, len(self.graph_simple.nodes))
        flat = [node for _,members in communities.items() for node in members]
        self.assertEqual(len(flat), len(set(flat)))

if __name__ == "__main__":
    unittest.main()
