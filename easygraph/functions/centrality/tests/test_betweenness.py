import unittest

import easygraph as eg


class Test_betweenness(unittest.TestCase):
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

    def test_betweenness(self):
        for i in self.test_graphs:
            print(eg.functions.betweenness_centrality(i))


if __name__ == "__main__":
    unittest.main()
