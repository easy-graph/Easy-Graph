import unittest

import easygraph as eg


class Test_pagerank(unittest.TestCase):
    def setUp(self):
        edges = [
            (1, 2),
            (2, 3),
            ("String", "Bool"),
            (2, 1),
            (0, 0),
            ((None, None), (None, None)),
        ]
        self.g = eg.classes.DiGraph(edges)

    def test_pagerank(self):
        test_graphs = [eg.Graph(), eg.DiGraph()]
        for i in test_graphs:
            print(eg.functions.pagerank(i))

        print(self.g.nodes)
        print(eg.functions.pagerank(self.g))

    """
    def test_google_matrix(self):
        test_graphs = [eg.Graph(), eg.DiGraph(), eg.MultiGraph(), eg.MultiDiGraph()]
        for g in test_graphs:
            print(eg.functions.pagerank.(g))
    """


if __name__ == "__main__":
    unittest.main()
