import unittest

import easygraph as eg


class Test_closeness(unittest.TestCase):
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

    def test_closeness(self):
        for i in self.test_graphs:
            print(i.nodes)
            print(eg.functions.closeness_centrality(i))


if __name__ == "__main__":
    unittest.main()
