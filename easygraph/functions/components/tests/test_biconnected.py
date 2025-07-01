import unittest

import easygraph as eg
import pytest

from easygraph import biconnected_components
from easygraph import generator_articulation_points
from easygraph import generator_biconnected_components_edges
from easygraph import generator_biconnected_components_nodes
from easygraph import is_biconnected


class Test_biconnected(unittest.TestCase):
    def setUp(self):
        self.edges = [(1, 2), (2, 3), ("String", "Bool"), (2, 1), (0, 0), (-99, 256)]
        self.test_graphs = [eg.Graph(), eg.MultiGraph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

    def test_is_biconnected(self):
        for i in self.test_graphs:
            print(eg.is_biconnected(i))

    def test_biconnected_components(self):
        for i in self.test_graphs:
            eg.biconnected_components(i)

    def test_generator_biconnected_components_nodes(self):
        for i in self.test_graphs:
            eg.generator_biconnected_components_nodes(i)

    def test_generator_biconnected_components_edges(self):
        for i in self.test_graphs:
            eg.generator_biconnected_components_edges(i)

    def test_generator_articulation_points(self):
        for i in self.test_graphs:
            eg.generator_articulation_points(i)


class TestBiconnectedFunctions(unittest.TestCase):
    def test_single_node(self):
        G = eg.Graph()
        G.add_node(1)
        self.assertFalse(is_biconnected(G))
        self.assertEqual(list(biconnected_components(G)), [])
        self.assertEqual(list(generator_articulation_points(G)), [])

    def test_disconnected_graph(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        self.assertFalse(is_biconnected(G))
        self.assertGreaterEqual(len(list(generator_biconnected_components_edges(G))), 1)

    def test_triangle(self):
        G = eg.Graph([(0, 1), (1, 2), (2, 0)])
        self.assertTrue(is_biconnected(G))
        comps = list(biconnected_components(G))
        self.assertEqual(len(comps), 1)
        self.assertEqual(set(comps[0]), {(0, 1), (1, 2), (2, 0)})
        self.assertEqual(list(generator_articulation_points(G)), [])

    def test_with_articulation_point(self):
        G = eg.Graph([(0, 1), (1, 2), (1, 3)])
        self.assertFalse(is_biconnected(G))
        arts = list(generator_articulation_points(G))
        self.assertIn(1, arts)
        self.assertEqual(len(arts), 1)

    def test_cycle_plus_leaf(self):
        G = eg.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
        self.assertFalse(is_biconnected(G))
        arts = list(generator_articulation_points(G))
        self.assertIn(2, arts)

    def test_multiple_biconnected_components(self):
        G = eg.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])  # triangle
        G.add_edges_from([(3, 4), (4, 5)])  # path
        components = list(generator_biconnected_components_edges(G))
        self.assertEqual(len(components), 3)
        nodes_comps = list(generator_biconnected_components_nodes(G))
        self.assertTrue(any({1, 2, 3}.issubset(comp) for comp in nodes_comps))
        self.assertTrue(any({4, 5}.issubset(comp) for comp in nodes_comps))

    def test_articulation_points_multiple(self):
        G = eg.Graph([(0, 1), (1, 2), (2, 3), (3, 4)])
        aps = list(generator_articulation_points(G))
        self.assertEqual(aps, [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
