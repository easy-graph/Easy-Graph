import json

import easygraph as eg
import pytest

from easygraph.readwrite.json_graph import adjacency_data
from easygraph.readwrite.json_graph import adjacency_graph


class TestAdjacency:
    def test_graph(self):
        G = eg.path_graph(4)
        H = adjacency_graph(adjacency_data(G))
        eg.is_isomorphic(G, H)

    def test_graph_attributes(self):
        G = eg.path_graph(4)
        G.add_node(1, color="red")
        G.add_edge(1, 2, width=7)
        G.graph["foo"] = "bar"
        G.graph[1] = "one"

        H = adjacency_graph(adjacency_data(G))
        assert H.graph["foo"] == "bar"
        assert H.nodes[1]["color"] == "red"
        assert H[1][2]["width"] == 7

        d = json.dumps(adjacency_data(G))
        H = adjacency_graph(json.loads(d))
        assert H.graph["foo"] == "bar"
        assert H.graph[1] == "one"
        assert H.nodes[1]["color"] == "red"
        assert H[1][2]["width"] == 7

    def test_digraph(self):
        G = eg.DiGraph()
        eg.add_path(G, [1, 2, 3])
        H = adjacency_graph(adjacency_data(G))
        assert H.is_directed()
        eg.is_isomorphic(G, H)

    def test_multidigraph(self):
        G = eg.MultiDiGraph()
        eg.add_path(G, [1, 2, 3])
        H = adjacency_graph(adjacency_data(G))
        assert H.is_directed()
        assert H.is_multigraph()

    def test_multigraph(self):
        G = eg.MultiGraph()
        G.add_edge(1, 2, key="first")
        G.add_edge(1, 2, key="second", color="blue")
        H = adjacency_graph(adjacency_data(G))
        eg.is_isomorphic(G, H)
        assert H[1][2]["second"]["color"] == "blue"

    def test_exception(self):
        with pytest.raises(eg.EasyGraphError):
            G = eg.MultiDiGraph()
            attrs = dict(id="node", key="node")
            adjacency_data(G, attrs)
