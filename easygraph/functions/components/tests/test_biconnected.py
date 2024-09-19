import unittest

import easygraph as eg


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


if __name__ == "__main__":
    unittest.main()
