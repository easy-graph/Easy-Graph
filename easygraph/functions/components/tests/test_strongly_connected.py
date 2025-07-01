import inspect
import unittest

import easygraph as eg

from easygraph import condensation
from easygraph import is_strongly_connected
from easygraph import number_strongly_connected_components
from easygraph import strongly_connected_components
from easygraph.utils.exception import EasyGraphNotImplemented
from easygraph.utils.exception import EasyGraphPointlessConcept


class Test_strongly_connected(unittest.TestCase):
    def setUp(self):
        self.edges = [(1, 2), (2, 3), ("String", "Bool"), (2, 1), (0, 0), (-99, 256)]
        self.test_graphs = [eg.Graph([(4, -4)]), eg.DiGraph([(4, False)])]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

    def test_empty_graph(self):
        G = eg.DiGraph()
        with self.assertRaises(EasyGraphPointlessConcept):
            is_strongly_connected(G)
        self.assertEqual(number_strongly_connected_components(G), 0)
        self.assertEqual(list(strongly_connected_components(G)), [])

    def test_single_node(self):
        G = eg.DiGraph()
        G.add_node(1)
        self.assertTrue(is_strongly_connected(G))
        self.assertEqual(number_strongly_connected_components(G), 1)
        scc = list(strongly_connected_components(G))
        self.assertEqual(scc, [{1}])

    def test_cycle_graph(self):
        G = eg.DiGraph([(1, 2), (2, 3), (3, 1)])
        self.assertTrue(is_strongly_connected(G))
        self.assertEqual(number_strongly_connected_components(G), 1)
        scc = list(strongly_connected_components(G))
        self.assertEqual(scc, [{1, 2, 3}])

    def test_disconnected_scc(self):
        G = eg.DiGraph([(0, 1), (1, 0), (2, 3), (3, 2), (4, 5)])
        scc = list(strongly_connected_components(G))
        self.assertEqual(len(scc), 4)
        self.assertIn({0, 1}, scc)
        self.assertIn({2, 3}, scc)
        self.assertIn({4}, scc)
        self.assertIn({5}, scc)
        self.assertFalse(is_strongly_connected(G))
        self.assertEqual(number_strongly_connected_components(G), 4)

    def test_scc_with_self_loops(self):
        G = eg.DiGraph([(1, 1), (2, 2), (3, 4), (4, 3)])
        scc = list(strongly_connected_components(G))
        self.assertEqual(len(scc), 3)
        self.assertIn({1}, scc)
        self.assertIn({2}, scc)
        self.assertIn({3, 4}, scc)

    def test_condensation_structure(self):
        G = eg.DiGraph(
            [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (3, 4), (5, 6), (6, 3), (6, 7)]
        )
        cond = condensation(G)
        self.assertTrue(cond.is_directed())
        self.assertIn("mapping", cond.graph)
        self.assertEqual(len(cond), number_strongly_connected_components(G))

        def has_cycle(G):
            visited = set()
            temp_mark = set()

            def visit(node):
                if node in temp_mark:
                    return True
                if node in visited:
                    return False
                temp_mark.add(node)
                for neighbor in G[node]:
                    if visit(neighbor):
                        return True
                temp_mark.remove(node)
                visited.add(node)
                return False

            return any(visit(v) for v in G)

        self.assertFalse(has_cycle(cond))

    def test_condensation_empty_graph(self):
        G = eg.DiGraph()
        C = condensation(G)
        self.assertEqual(len(C), 0)

    def test_undirected_raises(self):
        G = eg.Graph([(1, 2), (2, 3)])
        with self.assertRaises(EasyGraphNotImplemented):
            list(strongly_connected_components(G))
        with self.assertRaises(EasyGraphNotImplemented):
            is_strongly_connected(G)
        with self.assertRaises(EasyGraphNotImplemented):
            number_strongly_connected_components(G)

    def test_condensation_on_undirected_graph_raises(self):
        G = eg.Graph([(1, 2), (2, 3)])
        with self.assertRaises(EasyGraphNotImplemented):
            condensation(G)

    def test_condensation_manual_scc_input(self):
        G = eg.DiGraph([(1, 2), (2, 1), (3, 4)])
        scc = list(strongly_connected_components(G))
        C = condensation(G, scc=scc)
        self.assertEqual(len(C.nodes), len(scc))
        # Check if mapping is consistent
        all_mapped = set(C.graph["mapping"].keys())
        self.assertEqual(all_mapped, set(G.nodes))


if __name__ == "__main__":
    unittest.main()
