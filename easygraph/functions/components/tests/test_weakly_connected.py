import unittest

import easygraph as eg

from easygraph import is_weakly_connected
from easygraph import number_weakly_connected_components
from easygraph import weakly_connected_components
from easygraph.utils.exception import EasyGraphNotImplemented
from easygraph.utils.exception import EasyGraphPointlessConcept


class Test_weakly_connected(unittest.TestCase):
    def test_empty_graph(self):
        G = eg.DiGraph()
        with self.assertRaises(EasyGraphPointlessConcept):
            is_weakly_connected(G)
        self.assertEqual(number_weakly_connected_components(G), 0)
        self.assertEqual(list(weakly_connected_components(G)), [])

    def test_single_node(self):
        G = eg.DiGraph()
        G.add_node(1)
        self.assertTrue(is_weakly_connected(G))
        self.assertEqual(number_weakly_connected_components(G), 1)
        self.assertEqual(list(weakly_connected_components(G)), [{1}])

    def test_connected_graph(self):
        G = eg.DiGraph([(1, 2), (2, 3), (3, 4)])
        self.assertTrue(is_weakly_connected(G))
        self.assertEqual(number_weakly_connected_components(G), 1)
        self.assertEqual(list(weakly_connected_components(G)), [{1, 2, 3, 4}])

    def test_disconnected_graph(self):
        G = eg.DiGraph([(1, 2), (3, 4)])
        self.assertFalse(is_weakly_connected(G))
        wcc = list(weakly_connected_components(G))
        self.assertEqual(len(wcc), 2)
        self.assertIn({1, 2}, wcc)
        self.assertIn({3, 4}, wcc)

    def test_self_loops(self):
        G = eg.DiGraph([(1, 1), (2, 2)])
        wcc = list(weakly_connected_components(G))
        self.assertEqual(len(wcc), 2)
        self.assertIn({1}, wcc)
        self.assertIn({2}, wcc)
        self.assertFalse(is_weakly_connected(G))

    def test_multiple_components(self):
        G = eg.DiGraph([(1, 2), (3, 4), (5, 6), (6, 5)])
        wcc = list(weakly_connected_components(G))
        self.assertEqual(number_weakly_connected_components(G), 3)
        self.assertIn({1, 2}, wcc)
        self.assertIn({3, 4}, wcc)
        self.assertIn({5, 6}, wcc)

    def test_unconnected_nodes(self):
        G = eg.DiGraph([(1, 2), (3, 4)])
        G.add_node(99)  # isolated node
        wcc = list(weakly_connected_components(G))
        self.assertEqual(len(wcc), 3)
        self.assertIn({99}, wcc)

    def test_is_weakly_connected_after_adding_edge(self):
        G = eg.DiGraph([(0, 1), (2, 1)])
        G.add_node(3)
        self.assertFalse(is_weakly_connected(G))
        G.add_edge(2, 3)
        self.assertTrue(is_weakly_connected(G))

    def test_undirected_raises(self):
        G = eg.Graph([(1, 2), (2, 3)])
        with self.assertRaises(EasyGraphNotImplemented):
            is_weakly_connected(G)
        with self.assertRaises(EasyGraphNotImplemented):
            number_weakly_connected_components(G)
        with self.assertRaises(EasyGraphNotImplemented):
            list(weakly_connected_components(G))


if __name__ == "__main__":
    unittest.main()
