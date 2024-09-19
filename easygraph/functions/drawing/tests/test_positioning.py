import unittest

import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np


class TestPositioning(unittest.TestCase):
    def setUp(self):
        self.ds = eg.datasets.get_graph_karateclub()
        self.edges = [
            (1, 4),
            (2, 4),
            ("String", "Bool"),
            (4, 1),
            (0, 4),
            (4, 256),
            ((1, 2), (3, 4)),
        ]
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))
        self.shs = eg.common_greedy(self.ds, int(len(self.ds.nodes) / 3))

    def test_random_position(self):
        print()
        for i in self.test_graphs:
            print(eg.random_position(i))

    def test_circular_position(self):
        print()
        for i in self.test_graphs:
            print(eg.circular_position(i))

    def test_shell_position(self):
        print()
        for i in self.test_graphs:
            print(eg.shell_position(i))

    def test_rescale_position(self):
        print()
        for i in self.test_graphs:
            try:
                pos = eg.random_position(i)
                obj = np.array(list(pos.values()))
                print(eg.rescale_position(obj))
            except Exception as e:
                print(e)

    def test_kamada_kawai_layout(self):
        print()
        for i in self.test_graphs:
            print(eg.kamada_kawai_layout(i))


if __name__ == "__main__":
    unittest.main()
