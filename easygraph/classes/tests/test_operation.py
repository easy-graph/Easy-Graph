import easygraph as eg
import pytest

from easygraph.classes import operation
from easygraph.utils import edges_equal


@pytest.mark.parametrize(
    "graph_type", [eg.Graph, eg.DiGraph, eg.MultiGraph, eg.MultiDiGraph]
)
def test_selfloops(graph_type):
    G = eg.complete_graph(3, create_using=graph_type)
    G.add_edge(0, 0)
    assert edges_equal(eg.selfloop_edges(G), [(0, 0)])
    assert edges_equal(eg.selfloop_edges(G, data=True), [(0, 0, {})])
    assert eg.number_of_selfloops(G) == 1


def test_set_edge_attributes_scalar():
    G = eg.path_graph(3)
    eg.set_edge_attributes(G, 5, "weight")
    for _, _, data in G.edges:
        assert data["weight"] == 5


def test_set_edge_attributes_dict():
    G = eg.path_graph(3)
    attrs = {(0, 1): 3, (1, 2): 7}
    eg.set_edge_attributes(G, attrs, "weight")
    assert G[0][1]["weight"] == 3
    assert G[1][2]["weight"] == 7


def test_set_edge_attributes_dict_of_dict():
    G = eg.path_graph(3)
    attrs = {(0, 1): {"a": 1}, (1, 2): {"b": 2}}
    eg.set_edge_attributes(G, attrs)
    assert G[0][1]["a"] == 1
    assert G[1][2]["b"] == 2


def test_set_node_attributes_scalar():
    G = eg.path_graph(3)
    eg.set_node_attributes(G, 42, "level")
    for n in G.nodes:
        assert G.nodes[n]["level"] == 42


def test_set_node_attributes_dict():
    G = eg.path_graph(3)
    eg.set_node_attributes(G, {0: "x", 1: "y"}, name="tag")
    assert G.nodes[0]["tag"] == "x"
    assert G.nodes[1]["tag"] == "y"


def test_set_node_attributes_dict_of_dict():
    G = eg.path_graph(3)
    eg.set_node_attributes(G, {0: {"foo": 10}, 1: {"bar": 20}})
    assert G.nodes[0]["foo"] == 10
    assert G.nodes[1]["bar"] == 20


def test_add_path_structure_and_attrs():
    G = eg.Graph()
    eg.add_path(G, [10, 11, 12], weight=9)
    actual_edges = {(u, v) for u, v, _ in G.edges}
    assert actual_edges == {(10, 11), (11, 12)}
    assert G[10][11]["weight"] == 9
    assert G[11][12]["weight"] == 9


def test_topological_sort_linear():
    G = eg.DiGraph()
    G.add_edges_from([(1, 2), (2, 3)])
    assert list(operation.topological_sort(G)) == [1, 2, 3]


def test_topological_sort_cycle():
    G = eg.DiGraph([(0, 1), (1, 2), (2, 0)])
    with pytest.raises(AssertionError, match="contains a cycle"):
        list(operation.topological_sort(G))


def test_selfloop_edges_variants():
    G = eg.MultiGraph()
    G.add_edge(0, 0, key="x", label="loop")
    G.add_edge(1, 1, key="y", label="loop2")
    basic = list(eg.selfloop_edges(G))
    with_data = list(eg.selfloop_edges(G, data=True))
    with_keys = list(eg.selfloop_edges(G, keys=True))
    full = list(eg.selfloop_edges(G, keys=True, data="label"))
    assert (0, 0) in basic and (1, 1) in basic
    assert all(len(t) == 3 for t in with_data)
    assert all(len(t) == 3 for t in with_keys)
    assert "x" in [k for _, _, k, _ in full]


def test_number_of_selfloops():
    G = eg.MultiGraph()
    G.add_edges_from([(0, 0), (1, 1), (1, 2)])
    assert eg.number_of_selfloops(G) == 2


def test_density_undirected():
    G = eg.complete_graph(5)
    d = eg.density(G)
    assert pytest.approx(d, 0.01) == 1.0


def test_density_directed():
    G = eg.DiGraph()
    G.add_edges_from([(0, 1), (1, 2)])
    d = eg.density(G)
    assert pytest.approx(d, 0.01) == 2 / (3 * (3 - 1))  # 2/6


def test_topological_generations_linear():
    G = eg.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    generations = list(operation.topological_generations(G))
    assert generations == [[1], [2], [3], [4]]


def test_topological_generations_branching():
    G = eg.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    generations = list(operation.topological_generations(G))
    # Valid topological generations: [1], [2, 3], [4]
    assert generations[0] == [1]
    assert set(generations[1]) == {2, 3}
    assert generations[2] == [4]
