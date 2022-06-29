import pytest

import easygraph as eg

class TestClustering:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("numpy")

    def test_clustering(self):
        G = eg.Graph()
        assert list(eg.clustering(G).values()) == []
        assert eg.clustering(G) == {}

    def test_path(self):
        G = eg.path_graph(10)
        assert list(eg.clustering(G).values()) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert eg.clustering(G) == {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
        }

    def test_k5(self):
        G = eg.complete_graph(5)
        assert list(eg.clustering(G).values()) == [1, 1, 1, 1, 1]
        G.remove_edge(1, 2)
        assert list(eg.clustering(G).values()) == [
            5 / 6,
            1,
            1,
            5 / 6,
            5 / 6,
        ]
        assert eg.clustering(G, [1, 4]) == {1: 1, 4: 0.83333333333333337}

    def test_k5_signed(self):
        G = eg.complete_graph(5)
        assert list(eg.clustering(G).values()) == [1, 1, 1, 1, 1]
        G.remove_edge(1, 2)
        G.add_edge(0, 1, weight=-1)
        assert list(eg.clustering(G, weight="weight").values()) == [
            1 / 6,
            -1 / 3,
            1,
            3 / 6,
            3 / 6,
        ]
