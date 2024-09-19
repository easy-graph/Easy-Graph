import unittest

import easygraph as eg


class test_classic(unittest.TestCase):
    def setUp(self):
        self.G = eg.datasets.get_graph_karateclub()

    def test_empty_graph(self):
        # print(eg.empty_graph(-1).nodes)
        print(eg.empty_graph(10).nodes)

    def test_path_graph(self):
        eg.path_graph(10, eg.DiGraph)

    def test_complete_graph(self):
        eg.complete_graph(10, eg.DiGraph)


if __name__ == "__main__":
    unittest.main()
