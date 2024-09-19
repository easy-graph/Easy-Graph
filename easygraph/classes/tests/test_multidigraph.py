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


if __name__ == "__main__":
    unittest.main()
# test()
