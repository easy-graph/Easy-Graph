import inspect
import unittest

import easygraph as eg


class TestConnected(unittest.TestCase):
    def setUp(self):
        self.edges = [(1, 2), (2, 3), ("String", "Bool"), (2, 1), (0, 0), (-99, 256)]
        self.test_graphs = [eg.Graph([(4, -4)]), eg.DiGraph([(4, False)])]
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


if __name__ == "__main__":
    unittest.main()
