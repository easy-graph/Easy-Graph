import unittest

import easygraph as eg


class test_random_network(unittest.TestCase):
    def setUp(self):
        self.G = eg.datasets.get_graph_karateclub()

    def test_erdos_renyi_M(self):
        print(eg.erdos_renyi_M(8, 5).edges)

    def test_erdos_renyi_P(self):
        print(eg.erdos_renyi_P(8, 0.2).nodes)

    def test_fast_erdos_renyi_P(self):
        print(eg.fast_erdos_renyi_P(8, 0.2).nodes)

    def test_WS_Random(self):
        print(eg.WS_Random(8, 1, 0.5).nodes)

    def test_graph_Gnm(self):
        print(eg.graph_Gnm(8, 5).nodes)


if __name__ == "__main__":
    unittest.main()
