import unittest

import easygraph as eg


class test_hypergraph_operation(unittest.TestCase):
    def setUp(self):
        self.g = eg.get_graph_karateclub()
        self.edges = [(1, 2), (8, 4)]
        self.hg = [
            eg.Hypergraph(num_v=10, e_list=self.edges, e_property=None),
            eg.Hypergraph(num_v=2, e_list=[(0, 1)]),
        ]

    def test_hypergraph_clustering_coefficient(self):
        for i in self.hg:
            print(eg.hypergraph_clustering_coefficient(i))

    def test_hypergraph_local_clustering_coefficient(self):
        for i in self.hg:
            print(eg.hypergraph_local_clustering_coefficient(i))

    def test_hypergraph_two_node_clustering_coefficient(self):
        for i in self.hg:
            print(eg.hypergraph_two_node_clustering_coefficient(i))


if __name__ == "__main__":
    unittest.main()
