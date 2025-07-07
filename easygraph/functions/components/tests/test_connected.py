import inspect
import unittest

import easygraph as eg

from easygraph import connected_component_of_node
from easygraph import connected_components
from easygraph import connected_components_directed
from easygraph import is_connected
from easygraph import number_connected_components
from easygraph.utils.exception import EasyGraphNotImplemented


class TestConnected(unittest.TestCase):
    def setUp(self):
        self.edges = [(1, 2), (2, 3), (0, 4), (2, 1), (0, 0), (-99, 256)]
        self.test_graphs = [eg.Graph([(4, -4)]), eg.DiGraph([(4, -4)])]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

    def test_is_connected(self):
        for i in self.test_graphs:
            print(eg.is_connected(i))

    def test_number_connected_components(self):
        for i in self.test_graphs:
            print(eg.number_connected_components(i))

    def test_connected_components(self):
        for i in self.test_graphs:
            print(eg.connected_components(i))

    def test_connected_components_directed(self):
        for i in self.test_graphs:
            print(eg.connected_components_directed(i))

    def test_connected_component_of_node(self):
        for i in self.test_graphs:
            print(eg.connected_component_of_node(i, 4))

    def test_empty_graph(self):
        G = eg.Graph()
        with self.assertRaises(AssertionError):
            is_connected(G)
        self.assertEqual(number_connected_components(G), 0)
        self.assertEqual(list(connected_components(G)), [])

    def test_single_node(self):
        G = eg.Graph()
        G.add_node(1)
        self.assertTrue(is_connected(G))
        self.assertEqual(number_connected_components(G), 1)
        self.assertEqual(list(connected_components(G)), [{1}])
        self.assertEqual(connected_component_of_node(G, 1), {1})

    def test_disconnected_graph(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        self.assertFalse(is_connected(G))
        self.assertEqual(number_connected_components(G), 2)
        comps = list(connected_components(G))
        self.assertTrue({0, 1} in comps and {2, 3} in comps)

    def test_connected_graph(self):
        G = eg.path_graph(5)
        self.assertTrue(is_connected(G))
        self.assertEqual(number_connected_components(G), 1)
        comps = list(connected_components(G))
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0], set(range(5)))

    def test_node_component_lookup(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        comp = connected_component_of_node(G, 0)
        self.assertEqual(comp, {0, 1})
        with self.assertRaises(KeyError):
            connected_component_of_node(G, 999)  # non-existent node

    def test_undirected_with_self_loops(self):
        G = eg.Graph()
        G.add_edges_from([(1, 1), (2, 2), (1, 2)])
        self.assertTrue(is_connected(G))
        self.assertEqual(number_connected_components(G), 1)
        self.assertEqual(list(connected_components(G))[0], {1, 2})

    def test_directed_components(self):
        G = eg.DiGraph()
        G.add_edges_from([(0, 1), (2, 3)])
        self.assertEqual(number_connected_components(G), 2)
        components = list(connected_components_directed(G))
        self.assertTrue({0, 1} in components and {2, 3} in components)

    def test_directed_strong_vs_weak(self):
        G = eg.DiGraph([(0, 1), (1, 0), (2, 3)])
        comps = list(connected_components_directed(G))
        self.assertTrue({0, 1} in comps)
        self.assertTrue({2, 3} in comps)

    def test_multigraph_blocked(self):
        G = eg.MultiGraph([(1, 2), (2, 3)])
        with self.assertRaises(EasyGraphNotImplemented):
            is_connected(G)
        with self.assertRaises(EasyGraphNotImplemented):
            number_connected_components(G)
        with self.assertRaises(EasyGraphNotImplemented):
            list(connected_components(G))
        with self.assertRaises(EasyGraphNotImplemented):
            connected_component_of_node(G, 1)


if __name__ == "__main__":
    unittest.main()
