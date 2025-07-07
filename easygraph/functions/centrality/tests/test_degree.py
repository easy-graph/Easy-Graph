import unittest

import easygraph as eg

from easygraph.utils.exception import EasyGraphNotImplemented


class Test_degree(unittest.TestCase):
    def setUp(self):
        self.edges = [
            (1, 4),
            (2, 4),
            ("String", "Bool"),
            (4, 1),
            (0, 4),
            (4, 256),
            ((None, None), (None, None)),
        ]
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

        self.undirected_graph = eg.Graph()
        self.undirected_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        # Directed graph
        self.directed_graph = eg.DiGraph()
        self.directed_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        # Single-node graph
        self.single_node_graph = eg.Graph()
        self.single_node_graph.add_node(0)

        # Empty graph
        self.empty_graph = eg.Graph()

        # Multigraph
        self.multigraph = eg.MultiGraph()
        self.multigraph.add_edges_from([(0, 1), (0, 1)])

    def test_degree(self):
        for i in self.test_graphs:
            print(i.edges)
            print(eg.functions.degree_centrality(i))
            print(eg.functions.in_degree_centrality(i))
            print(eg.functions.out_degree_centrality(i))

    def test_degree_centrality_undirected(self):
        result = eg.functions.degree_centrality(self.undirected_graph)
        self.assertEqual(len(result), len(self.undirected_graph))
        self.assertTrue(all(isinstance(v, float) for v in result.values()))

    def test_degree_centrality_directed(self):
        result = eg.functions.degree_centrality(self.directed_graph)
        self.assertEqual(len(result), len(self.directed_graph))

    def test_degree_centrality_single_node(self):
        result = eg.functions.degree_centrality(self.single_node_graph)
        self.assertEqual(result, {0: 1})

    def test_degree_centrality_empty_graph(self):
        result = eg.functions.degree_centrality(self.empty_graph)
        self.assertEqual(result, {})

    def test_in_out_degree_centrality_directed(self):
        in_deg = eg.functions.in_degree_centrality(self.directed_graph)
        out_deg = eg.functions.out_degree_centrality(self.directed_graph)
        self.assertEqual(len(in_deg), len(self.directed_graph))
        self.assertEqual(len(out_deg), len(self.directed_graph))

    def test_in_out_degree_centrality_single_node(self):
        G = eg.DiGraph()
        G.add_node(1)
        self.assertEqual(eg.functions.in_degree_centrality(G), {1: 1})
        self.assertEqual(eg.functions.out_degree_centrality(G), {1: 1})


if __name__ == "__main__":
    unittest.main()
