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

    def test_basic_predecessor(self):
        G = eg.path_graph(4)
        result = eg.predecessor(G, 0)
        assert result == {0: [], 1: [0], 2: [1], 3: [2]}

    def test_with_return_seen(self):
        G = eg.path_graph(4)
        pred, seen = eg.predecessor(G, 0, return_seen=True)
        assert pred == {0: [], 1: [0], 2: [1], 3: [2]}
        assert seen == {0: 0, 1: 1, 2: 2, 3: 3}

    def test_with_target(self):
        G = eg.path_graph(4)
        assert eg.predecessor(G, 0, target=2) == [1]

    def test_with_target_and_return_seen(self):
        G = eg.path_graph(4)
        pred, seen = eg.predecessor(G, 0, target=2, return_seen=True)
        assert pred == [1]
        assert seen == 2

    def test_with_cutoff(self):
        G = eg.path_graph(4)
        pred = eg.predecessor(G, 0, cutoff=1)
        assert pred == {0: [], 1: [0]}

    def test_disconnected_graph(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        pred = eg.predecessor(G, 0)
        assert 2 not in pred and 3 not in pred

    def test_invalid_source(self):
        G = eg.path_graph(4)
        with pytest.raises(eg.NodeNotFound):
            eg.predecessor(G, 99)

    def test_no_path_to_target(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        assert eg.predecessor(G, 0, target=3) == []

    def test_no_path_to_target_with_return_seen(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        pred, seen = eg.predecessor(G, 0, target=3, return_seen=True)
        assert pred == []
        assert seen == -1

    def test_cycle_graph(self):
        G = eg.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # cycled graph
        pred = eg.predecessor(G, 0)
        assert set(pred.keys()) == set(G.nodes)

    def test_directed_graph(self):
        G = eg.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        pred = eg.predecessor(G, 0)
        assert pred == {0: [], 1: [0], 2: [1], 3: [2]}
