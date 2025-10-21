import easygraph as eg
import pytest


class TestMultiGraph:
    def setup_method(self):
        self.Graph = eg.MultiGraph
        # build K3
        ed1, ed2, ed3 = ({0: {}}, {0: {}}, {0: {}})
        self.k3adj = {0: {1: ed1, 2: ed2}, 1: {0: ed1, 2: ed3}, 2: {0: ed2, 1: ed3}}
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._adj = self.k3adj
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}

    def test_data_input(self):
        G = self.Graph({1: [2], 2: [1]}, name="test")
        assert G.name == "test"
        expected = [(1, {2: {0: {}}}), (2, {1: {0: {}}})]
        assert sorted(G.adj.items()) == expected

    def test_has_edge(self):
        G = self.K3
        assert G.has_edge(0, 1)
        assert not G.has_edge(0, -1)
        assert G.has_edge(0, 1, 0)
        assert not G.has_edge(0, 1, 1)

    def test_get_edge_data(self):
        G = self.K3
        assert G.get_edge_data(0, 1) == {0: {}}
        assert G[0][1] == {0: {}}
        assert G[0][1][0] == {}
        assert G.get_edge_data(10, 20) is None
        assert G.get_edge_data(0, 1, 0) == {}

    def test_data_multigraph_input(self):
        # standard case with edge keys and edge data
        edata0 = dict(w=200, s="foo")
        edata1 = dict(w=201, s="bar")
        keydict = {0: edata0, 1: edata1}
        dododod = {"a": {"b": keydict}}

        multiple_edge = [("a", "b", 0, edata0), ("a", "b", 1, edata1)]
        single_edge = [("a", "b", 0, keydict)]

        G = self.Graph(dododod, multigraph_input=None)
        assert list(G.edges) == multiple_edge
        G = self.Graph(dododod, multigraph_input=False)
        assert list(G.edges) == single_edge

    def test_remove_node(self):
        G = self.K3
        G.remove_node(0)
        assert G.adj == {1: {2: {0: {}}}, 2: {1: {0: {}}}}
        with pytest.raises(eg.EasyGraphError):
            G.remove_node(-1)
