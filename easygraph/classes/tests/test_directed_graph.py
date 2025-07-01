import os
import unittest

from easygraph import DiGraph


class TestDiGraph(unittest.TestCase):
    def setUp(self):
        self.G = DiGraph()

    def test_add_node_and_exists(self):
        self.G.add_node("A")
        self.assertTrue(self.G.has_node("A"))
        self.assertIn("A", self.G.nodes)

    def test_add_nodes_with_attrs(self):
        self.G.add_nodes(["B", "C"], nodes_attr=[{"age": 30}, {"age": 40}])
        self.assertEqual(self.G.nodes["B"]["age"], 30)
        self.assertEqual(self.G.nodes["C"]["age"], 40)

    def test_add_edge_and_attrs(self):
        self.G.add_edge("A", "B", weight=5)
        self.assertTrue(self.G.has_edge("A", "B"))
        self.assertEqual(self.G.adj["A"]["B"]["weight"], 5)

    def test_add_edges_with_attrs(self):
        self.G.add_edges([("B", "C"), ("C", "D")], edges_attr=[{"w": 1}, {"w": 2}])
        self.assertEqual(self.G.adj["B"]["C"]["w"], 1)
        self.assertEqual(self.G.adj["C"]["D"]["w"], 2)

    def test_remove_node_and_edges(self):
        self.G.add_edges([("X", "Y"), ("Y", "Z")])
        self.G.remove_node("Y")
        self.assertFalse("Y" in self.G.nodes)
        self.assertFalse(self.G.has_edge("Y", "Z"))

    def test_remove_edge(self):
        self.G.add_edge("M", "N")
        self.G.remove_edge("M", "N")
        self.assertFalse(self.G.has_edge("M", "N"))

    def test_degrees(self):
        self.G.add_edges(
            [("A", "B"), ("C", "B")], edges_attr=[{"weight": 3}, {"weight": 2}]
        )

        in_degrees = self.G.in_degree(weight="weight")
        out_degrees = self.G.out_degree(weight="weight")
        degrees = self.G.degree(weight="weight")

        self.assertEqual(in_degrees["B"], 5)
        self.assertEqual(out_degrees["A"], 3)
        self.assertEqual(degrees["B"], 5)

    def test_neighbors_and_preds(self):
        self.G.add_edges([("P", "Q"), ("R", "P")])
        self.assertIn("Q", list(self.G.neighbors("P")))
        self.assertIn("R", list(self.G.predecessors("P")))
        all_n = list(self.G.all_neighbors("P"))
        self.assertIn("Q", all_n)
        self.assertIn("R", all_n)

    def test_size_and_num_edges_nodes(self):
        self.G.add_edges([("X", "Y"), ("Y", "Z")])
        self.assertEqual(self.G.size(), 2)
        self.assertEqual(self.G.number_of_edges(), 2)
        self.assertEqual(self.G.number_of_nodes(), 3)

    def test_subgraph_and_ego(self):
        self.G.add_edges([("A", "B"), ("B", "C"), ("C", "D")])
        sub = self.G.nodes_subgraph(["A", "B", "C"])
        self.assertTrue(sub.has_edge("A", "B"))
        self.assertFalse(sub.has_edge("C", "D"))
        ego = self.G.ego_subgraph("B")
        self.assertIn("A", ego.nodes or [])
        self.assertIn("C", ego.nodes or [])

    def test_to_index_node_graph(self):
        self.G.add_edges([("foo", "bar"), ("bar", "baz")])
        G2, node2idx, idx2node = self.G.to_index_node_graph()
        self.assertEqual(len(G2.nodes), 3)
        self.assertEqual(node2idx["foo"], 0)
        self.assertEqual(idx2node[0], "foo")

    def test_copy(self):
        self.G.add_edge("copyA", "copyB", weight=42)
        G_copy = self.G.copy()
        self.assertEqual(G_copy.adj["copyA"]["copyB"]["weight"], 42)

    def test_file_add_edges(self):
        fname = "temp_edges.txt"
        with open(fname, "w") as f:
            f.write("1 2 3.5\n2 3 4.5\n")
        self.G.add_edges_from_file(fname, weighted=True)
        os.remove(fname)
        self.assertEqual(self.G.adj["1"]["2"]["weight"], 3.5)
        self.assertEqual(self.G.adj["2"]["3"]["weight"], 4.5)
