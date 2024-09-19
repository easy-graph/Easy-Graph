import unittest

import easygraph as eg


class Test_flowbetweenness(unittest.TestCase):
    def setUp(self):
        self.edges = [
            (1, 2),
            (2, 3),
            ("String", "Bool"),
            (2, 1),
            (0, 0),
            (-99, 256),
            ((None, None), (None, None)),
        ]
        self.test_graphs = [eg.Graph(), eg.DiGraph()]
        self.test_graphs.append(eg.classes.DiGraph(self.edges))

    def test_flowbetweenness_centrality(self):
        for i in self.test_graphs:
            print(i.edges)
            print(eg.functions.flowbetweenness_centrality(i))


if __name__ == "__main__":
    unittest.main()
