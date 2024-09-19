import unittest

import easygraph as eg


class Test_egobetweenness(unittest.TestCase):
    def setUp(self):
        self.edges = [
            (1, 4),
            (2, 4),
            ("String", "Bool"),
            (4, 1),
            (0, 4),
            (4, 256),
            ((None, None), (None, None)),
        ]
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))
        print(self.test_graphs[-1].edges)

    def test_egobetweenness(self):
        print(eg.functions.ego_betweenness(self.test_graphs[-1], 4))


if __name__ == "__main__":
    unittest.main()
