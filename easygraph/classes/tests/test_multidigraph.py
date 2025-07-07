import unittest

import easygraph as eg
import pytest


class Test(unittest.TestCase):
    def setUp(self):
        edges = [(1, 2), (2, 3), ("String", "Bool"), (2, 1), ((1, 2), (3, 4))]
        self.g = eg.MultiDiGraph(edges)

    def test_add_edge(self):
        self.g.add_edge("from_Beijing", "to_California", key=3, attr=None)
        print(self.g.edges)

    def test_remove_edge(self):
        self.g.add_edge("from_Beijing", "to_California", key=3, attr=None)
        self.g.remove_edge("from_Beijing", "to_California")
        print(self.g.edges)

    def test_degree(self):
        print(self.g.degree)
        print(self.g.in_degree)
        print(self.g.out_degree)

    def test_reverse(self):
        # error with _succ
        print(self.g.reverse(copy=True).edges)
        # print(self.g.reverse(copy=False).edges)

    def test_attributes(self):
        print(self.g.edges)
        print(self.g.in_edges)


class TestMultiDiGraph(unittest.TestCase):
    def setUp(self):
        self.G = eg.MultiDiGraph()

    def test_add_edge_without_key(self):
        key1 = self.G.add_edge("A", "B", weight=1)
        key2 = self.G.add_edge("A", "B", weight=2)
        self.assertNotEqual(key1, key2)
        self.assertEqual(len(self.G._adj["A"]["B"]), 2)

    def test_add_edge_with_key(self):
        key = self.G.add_edge("A", "B", key="mykey", weight=3)
        self.assertEqual(key, "mykey")
        self.assertEqual(self.G._adj["A"]["B"]["mykey"]["weight"], 3)

    def test_edge_attributes_update(self):
        self.G.add_edge("X", "Y", key=1, color="red")
        self.G.add_edge("X", "Y", key=1, shape="circle")
        self.assertEqual(self.G._adj["X"]["Y"][1]["color"], "red")
        self.assertEqual(self.G._adj["X"]["Y"][1]["shape"], "circle")

    def test_remove_edge_by_key(self):
        self.G.add_edge("A", "B", key="k1")
        self.G.add_edge("A", "B", key="k2")
        self.G.remove_edge("A", "B", key="k1")
        self.assertIn("k2", self.G._adj["A"]["B"])
        self.assertNotIn("k1", self.G._adj["A"]["B"])

    def test_remove_edge_without_key(self):
        self.G.add_edge("A", "B", key="auto1")
        self.G.add_edge("A", "B", key="auto2")
        self.G.remove_edge("A", "B")
        # Only one of the keys should remain
        self.assertEqual(len(self.G._adj["A"]["B"]), 1)

    def test_remove_nonexistent_edge_raises(self):
        with self.assertRaises(eg.EasyGraphError):
            self.G.remove_edge("X", "Y", key="doesnotexist")

    def test_edges_property(self):
        self.G.add_edge("U", "V", key="k", weight=5)
        edges = self.G.edges
        self.assertIn(("U", "V", "k", {"weight": 5}), edges)

    def test_in_out_degree(self):
        self.G.add_edge("A", "B", weight=3)
        self.G.add_edge("C", "B", weight=2)

        in_deg = {}
        for n in self.G._node:
            preds = self.G._pred[n]
            in_deg[n] = sum(
                d.get("weight", 1)
                for key_dict in preds.values()
                for d in key_dict.values()
            )

        self.assertEqual(in_deg["B"], 5)

    def test_to_undirected(self):
        self.G.add_edge("A", "B", key="k", weight=10)
        UG = self.G.to_undirected()
        self.assertTrue(UG.has_edge("A", "B"))
        self.assertEqual(UG["A"]["B"]["k"]["weight"], 10)

    def test_reverse_graph(self):
        self.G.add_edge("A", "B", key="k", data=99)
        RG = self.G.reverse()
        self.assertTrue(RG.has_edge("B", "A"))
        self.assertEqual(RG["B"]["A"]["k"]["data"], 99)

    def test_is_multigraph_and_directed(self):
        self.assertTrue(self.G.is_multigraph())
        self.assertTrue(self.G.is_directed())


if __name__ == "__main__":
    unittest.main()
# test()
