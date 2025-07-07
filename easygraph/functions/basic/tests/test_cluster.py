import easygraph as eg
import pytest


class TestClustering:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("numpy")

    def test_clustering(self):
        G = eg.DiGraph()
        G.add_edge("1", "2", weight=16)
        G.add_edge("2", "3", weight=16)
        G.add_edge("4", "3", weight=16)
        G.add_edge("3", "4", weight=23)
        G.add_edge("3", "5", weight=16)
        G.add_edge("4", "2", weight=20)
        print("clustering" in dir(eg))
        assert eg.clustering(G) == {
            "1": 0,
            "2": 0.3333333333333333,
            "3": 0.2,
            "4": 0.5,
            "5": 0,
        }

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
        assert eg.average_clustering(G) == 1
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
        assert eg.average_clustering(G) == 1
        G.remove_edge(1, 2)
        G.add_edge(0, 1, weight=-1)
        assert list(eg.clustering(G, weight="weight").values()) == [
            1 / 6,
            -1 / 3,
            1,
            3 / 6,
            3 / 6,
        ]


class TestDirectedClustering:
    def test_clustering(self):
        G = eg.DiGraph()
        assert list(eg.clustering(G).values()) == []
        assert eg.clustering(G) == {}

    def test_path(self):
        G = eg.path_graph(10, create_using=eg.DiGraph())
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
        assert eg.clustering(G, 0) == 0

    def test_k5(self):
        G = eg.complete_graph(5, create_using=eg.DiGraph())
        assert list(eg.clustering(G).values()) == [1, 1, 1, 1, 1]
        assert eg.average_clustering(G) == 1
        G.remove_edge(1, 2)
        assert list(eg.clustering(G).values()) == [
            11 / 12,
            1,
            1,
            11 / 12,
            11 / 12,
        ]
        assert eg.clustering(G, [1, 4]) == {1: 1, 4: 11 / 12}
        G.remove_edge(2, 1)
        assert list(eg.clustering(G).values()) == [
            5 / 6,
            1,
            1,
            5 / 6,
            5 / 6,
        ]
        assert eg.clustering(G, [1, 4]) == {1: 1, 4: 0.83333333333333337}
        assert eg.clustering(G, 4) == 5 / 6

    def test_triangle_and_edge(self):
        G = eg.empty_graph(range(3), eg.DiGraph())
        G.add_edges_from(eg.pairwise(range(3), cyclic=True))
        G.add_edge(0, 4)
        assert eg.clustering(G)[0] == 1 / 6


class TestDirectedAverageClustering:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("numpy")

    def test_empty(self):
        G = eg.DiGraph()
        with pytest.raises(ZeroDivisionError):
            eg.average_clustering(G)

    def test_average_clustering(self):
        G = eg.empty_graph(range(3), eg.DiGraph())
        G.add_edges_from(eg.pairwise(range(3), cyclic=True))
        G.add_edge(2, 3)
        assert eg.average_clustering(G) == (1 + 1 + 1 / 3) / 8
        assert eg.average_clustering(G, count_zeros=True) == (1 + 1 + 1 / 3) / 8
        assert eg.average_clustering(G, count_zeros=False) == (1 + 1 + 1 / 3) / 6
        assert eg.average_clustering(G, [1, 2, 3]) == (1 + 1 / 3) / 6
        assert eg.average_clustering(G, [1, 2, 3], count_zeros=True) == (1 + 1 / 3) / 6
        assert eg.average_clustering(G, [1, 2, 3], count_zeros=False) == (1 + 1 / 3) / 4


class TestAverageClustering:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("numpy")

    def test_empty(self):
        G = eg.Graph()
        with pytest.raises(ZeroDivisionError):
            eg.average_clustering(G)

    def test_average_clustering(self):
        G = eg.complete_graph(3)
        G.add_edge(2, 3)

        assert eg.average_clustering(G) == (1 + 1 + 1 / 3) / 4
        assert eg.average_clustering(G, count_zeros=True) == (1 + 1 + 1 / 3) / 4
        assert eg.average_clustering(G, count_zeros=False) == (1 + 1 + 1 / 3) / 3
        assert eg.average_clustering(G, [1, 2, 3]) == (1 + 1 / 3) / 3
        assert eg.average_clustering(G, [1, 2, 3], count_zeros=True) == (1 + 1 / 3) / 3
        assert eg.average_clustering(G, [1, 2, 3], count_zeros=False) == (1 + 1 / 3) / 2

    def test_average_clustering_signed(self):
        G = eg.complete_graph(3)
        G.add_edge(2, 3)
        G.add_edge(0, 1, weight=-1)
        assert eg.average_clustering(G, weight="weight") == (-1 - 1 - 1 / 3) / 4
        assert (
            eg.average_clustering(G, weight="weight", count_zeros=True)
            == (-1 - 1 - 1 / 3) / 4
        )
        assert (
            eg.average_clustering(G, weight="weight", count_zeros=False)
            == (-1 - 1 - 1 / 3) / 3
        )


class TestDirectedWeightedClustering:
    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip("numpy")

    def test_clustering(self):
        G = eg.DiGraph()
        assert list(eg.clustering(G, weight="weight").values()) == []
        assert eg.clustering(G) == {}

    def test_path(self):
        G = eg.path_graph(10, create_using=eg.DiGraph())
        print("type:", eg.clustering(G, weight="weight"))
        assert list(eg.clustering(G, weight="weight").values()) == [
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
        assert eg.clustering(G, weight="weight") == {
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
        G = eg.complete_graph(5, create_using=eg.DiGraph())
        assert list(eg.clustering(G, weight="weight").values()) == [1, 1, 1, 1, 1]
        assert eg.average_clustering(G, weight="weight") == 1
        G.remove_edge(1, 2)
        assert list(eg.clustering(G, weight="weight").values()) == [
            11 / 12,
            1,
            1,
            11 / 12,
            11 / 12,
        ]
        assert eg.clustering(G, [1, 4], weight="weight") == {1: 1, 4: 11 / 12}
        G.remove_edge(2, 1)
        assert list(eg.clustering(G, weight="weight").values()) == [
            5 / 6,
            1,
            1,
            5 / 6,
            5 / 6,
        ]
        assert eg.clustering(G, [1, 4], weight="weight") == {
            1: 1,
            4: 0.83333333333333337,
        }

    def test_triangle_and_edge(self):
        G = eg.empty_graph(range(3), create_using=eg.DiGraph())
        G.add_edges_from(eg.pairwise(range(3), cyclic=True))
        G.add_edge(0, 4, weight=2)
        assert eg.clustering(G)[0] == 1 / 6
        # Relaxed comparisons to allow graphblas-algorithms to pass tests
        np.testing.assert_allclose(eg.clustering(G, weight="weight")[0], 1 / 12)
        np.testing.assert_allclose(eg.clustering(G, 0, weight="weight"), 1 / 12)


class TestWeightedClustering:
    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip("numpy")

    def test_clustering(self):
        G = eg.Graph()
        assert list(eg.clustering(G, weight="weight").values()) == []
        assert eg.clustering(G) == {}

    def test_path(self):
        G = eg.path_graph(10)
        assert list(eg.clustering(G, weight="weight").values()) == [
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
        assert eg.clustering(G, weight="weight") == {
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

    def test_cubical(self):
        G = eg.from_dict_of_lists(
            {
                0: [1, 3, 4],
                1: [0, 2, 7],
                2: [1, 3, 6],
                3: [0, 2, 5],
                4: [0, 5, 7],
                5: [3, 4, 6],
                6: [2, 5, 7],
                7: [1, 4, 6],
            },
            create_using=None,
        )
        assert list(eg.clustering(G, weight="weight").values()) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert eg.clustering(G, 1) == 0
        assert list(eg.clustering(G, [1, 2], weight="weight").values()) == [0, 0]
        assert eg.clustering(G, 1, weight="weight") == 0
        assert eg.clustering(G, [1, 2], weight="weight") == {1: 0, 2: 0}

    def test_k5(self):
        G = eg.complete_graph(5)
        assert list(eg.clustering(G, weight="weight").values()) == [1, 1, 1, 1, 1]
        assert eg.average_clustering(G, weight="weight") == 1
        G.remove_edge(1, 2)
        assert list(eg.clustering(G, weight="weight").values()) == [
            5 / 6,
            1,
            1,
            5 / 6,
            5 / 6,
        ]
        assert eg.clustering(G, [1, 4], weight="weight") == {
            1: 1,
            4: 0.83333333333333337,
        }

    def test_triangle_and_edge(self):
        G = eg.empty_graph(range(3), None)
        G.add_edges_from(eg.pairwise(range(3), cyclic=True))
        G.add_edge(0, 4, weight=2)
        assert eg.clustering(G)[0] == 1 / 3
        np.testing.assert_allclose(eg.clustering(G, weight="weight")[0], 1 / 6)
        np.testing.assert_allclose(eg.clustering(G, 0, weight="weight"), 1 / 6)

    def test_triangle_and_signed_edge(self):
        G = eg.empty_graph(range(3), None)
        G.add_edges_from(eg.pairwise(range(3), cyclic=True))
        G.add_edge(0, 1, weight=-1)
        G.add_edge(3, 0, weight=0)
        assert eg.clustering(G)[0] == 1 / 3
        assert eg.clustering(G, weight="weight")[0] == -1 / 3


class TestAdditionalClusteringCases:
    def test_self_loops_ignored(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        G.add_edge(0, 0)  # self-loop
        assert eg.clustering(G, 0) == 1.0

    def test_isolated_node(self):
        G = eg.Graph()
        G.add_node(1)
        assert eg.clustering(G) == {1: 0}

    def test_degree_one_node(self):
        G = eg.Graph()
        G.add_edge(1, 2)
        assert eg.clustering(G) == {1: 0, 2: 0}

    def test_custom_weight_name(self):
        G = eg.Graph()
        G.add_edge(0, 1, strength=2)
        G.add_edge(1, 2, strength=2)
        G.add_edge(2, 0, strength=2)
        result = eg.clustering(G, weight="strength")
        assert result[0] > 0

    def test_negative_weights_mixed(self):
        G = eg.complete_graph(3)
        G[0][1]["weight"] = -1
        G[1][2]["weight"] = 1
        G[2][0]["weight"] = 1
        assert eg.clustering(G, 0, weight="weight") < 0

    def test_directed_reciprocal_edges(self):
        G = eg.DiGraph()
        G.add_edges_from([(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)])
        result = eg.clustering(G)
        assert all(0 <= v <= 1 for v in result.values())
