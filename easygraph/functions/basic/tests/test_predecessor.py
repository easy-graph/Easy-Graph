import easygraph as eg
import pytest


class TestPredecessor:
    # @classmethod
    # def setup_class(self):
    #     pytest.importskip("numpy")

    def test_predecessor(self):
        G = eg.path_graph(4)
        for source in G:
            assert eg.predecessor(G, source) in [
                {0: [], 1: [0], 2: [1], 3: [2]},
                {1: [], 0: [1], 2: [1], 3: [2]},
                {2: [], 1: [2], 3: [2], 0: [1]},
                {3: [], 2: [3], 1: [2], 0: [1]},
            ]
