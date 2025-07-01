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


class TestMultiGraphExtended:
    def test_add_multiple_edges_and_keys(self):
        G = eg.MultiGraph()
        k0 = G.add_edge(1, 2)
        k1 = G.add_edge(1, 2)
        assert k0 == 0
        assert k1 == 1
        assert G.number_of_edges(1, 2) == 2

    def test_add_edge_with_key_and_attributes(self):
        G = eg.MultiGraph()
        k = G.add_edge(1, 2, key="custom", weight=3, label="test")
        assert k == "custom"
        assert G.get_edge_data(1, 2, "custom") == {"weight": 3, "label": "test"}

    def test_add_edges_from_various_formats(self):
        G = eg.MultiGraph()
        edges = [
            (1, 2),  # 2-tuple
            (2, 3, {"weight": 7}),  # 3-tuple with attr
            (3, 4, "k1", {"color": "red"}),  # 4-tuple
        ]
        keys = G.add_edges_from(edges, capacity=100)
        assert len(keys) == 3
        assert G.get_edge_data(3, 4, "k1")["color"] == "red"
        assert G.get_edge_data(2, 3, 0)["capacity"] == 100

    def test_remove_edge_with_key(self):
        G = eg.MultiGraph()
        G.add_edge(1, 2, key="a")
        G.add_edge(1, 2, key="b")
        G.remove_edge(1, 2, key="a")
        assert not G.has_edge(1, 2, key="a")
        assert G.has_edge(1, 2, key="b")

    def test_remove_edge_arbitrary(self):
        G = eg.MultiGraph()
        G.add_edge(1, 2)
        G.add_edge(1, 2)
        G.remove_edge(1, 2)
        assert G.number_of_edges(1, 2) == 1

    def test_remove_edges_from_mixed(self):
        G = eg.MultiGraph()
        keys = G.add_edges_from([(1, 2), (1, 2), (2, 3)])
        G.remove_edges_from([(1, 2), (2, 3)])
        assert G.number_of_edges(1, 2) == 1
        assert G.number_of_edges(2, 3) == 0

    def test_to_directed_graph(self):
        G = eg.MultiGraph()
        G.add_edge(0, 1, weight=10)
        D = G.to_directed()
        assert D.is_directed()
        assert D.has_edge(0, 1)
        assert D.has_edge(1, 0)
        assert D.get_edge_data(0, 1, 0)["weight"] == 10

    def test_copy_graph(self):
        G = eg.MultiGraph()
        G.add_edge(1, 2, key="x", weight=9)
        H = G.copy()
        assert H.get_edge_data(1, 2, "x") == {"weight": 9}
        assert H is not G
        assert H.get_edge_data(1, 2, "x") is not G.get_edge_data(1, 2, "x")

    def test_has_edge_variants(self):
        G = eg.MultiGraph()
        G.add_edge(1, 2)
        G.add_edge(1, 2, key="z")
        assert G.has_edge(1, 2)
        assert G.has_edge(1, 2, key="z")
        assert not G.has_edge(2, 1, key="nonexistent")

    def test_get_edge_data_defaults(self):
        G = eg.MultiGraph()
        assert G.get_edge_data(10, 20) is None
        assert G.get_edge_data(10, 20, key="any", default="missing") == "missing"

    def test_edge_property_returns_all_edges(self):
        G = eg.MultiGraph()
        G.add_edge(0, 1, key=5, label="important")
        G.add_edge(1, 0, key=3, label="also important")
        edges = list(G.edges)
        assert any((0, 1, 5, {"label": "important"}) == e for e in edges)
        assert any((0, 1, 3, {"label": "also important"}) == e for e in edges)
