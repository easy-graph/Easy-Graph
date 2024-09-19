import inspect
import unittest

import easygraph as eg


class Test_strongly_connected(unittest.TestCase):
    def setUp(self):
        self.edges = [(1, 2), (2, 3), ("String", "Bool"), (2, 1), (0, 0), (-99, 256)]
        self.test_graphs = [eg.Graph([(4, -4)]), eg.DiGraph([(4, False)])]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

    def test_number_strongly_connected_components(self):
        pass

    def test_strongly_connected_components(self):
        pass

    def test_is_strongly_connected(self):
        pass

    def test_condensation(self):
        pass


if __name__ == "__main__":
    unittest.main()
