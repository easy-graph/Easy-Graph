import easygraph as eg
import pytest

from easygraph import k_core


@pytest.mark.parametrize(
    "edges,k",
    [
        ([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)], 2),
        ([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)], 3),
        ([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], 2),
        ([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], 3),
        ([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], 1),
    ],
)
def test_k_core(edges, k):
    nx = pytest.importorskip("networkx")

    from easygraph import Graph
    from easygraph import k_core

    G = Graph()
    G_nx = nx.Graph()
    G.add_edges_from(edges)
    G_nx.add_edges_from(edges)

    H = k_core(G)
    H_nx = nx.core_number(G_nx)
    assert H == list(H_nx.values())


def test_k_core_empty_graph():
    G = eg.Graph()
    result = k_core(G)
    assert result == []


def test_k_core_single_node_isolated():
    G = eg.Graph()
    G.add_node(1)
    result = k_core(G)
    assert result == [0]


def test_k_core_clique():
    G = eg.complete_graph(5)  # Each node has degree 4
    result = k_core(G)
    assert set(result) == {4}


def test_k_core_star_graph():
    nx = pytest.importorskip("networkx")
    G = eg.Graph()
    G.add_node(0)
    G.add_edges_from((0, i) for i in range(1, 6))
    result = k_core(G)
    G_nx = nx.Graph()
    G_nx.add_node(0)
    G_nx.add_edges_from((0, i) for i in range(1, 6))
    expected = list(nx.core_number(G_nx).values())
    assert sorted(result) == sorted(expected)


def test_k_core_disconnected_components():
    G = eg.Graph()
    # Component 1: triangle
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Component 2: line
    G.add_edges_from([(3, 4)])
    result = k_core(G)
    core_component_1 = {result[i] for i in [0, 1, 2]}
    core_component_2 = {result[i] for i in [3, 4]}
    assert core_component_1 == {2}
    assert core_component_2 == {1}


def test_k_core_all_zero_core():
    G = eg.path_graph(5)
    result = k_core(G)
    assert all(isinstance(v, int) or isinstance(v, float) for v in result)
    assert max(result) <= 2


def test_k_core_index_to_node_mapping_consistency():
    G = eg.Graph()
    edges = [(5, 10), (10, 15), (15, 20)]
    G.add_edges_from(edges)
    result = k_core(G)
    for i, node in enumerate(G.index2node):
        assert isinstance(result[i], (int, float))
        deg_map = dict(G.degree())
        if node in deg_map:
            assert result[i] <= deg_map[node]


def test_k_core_large_k():
    G = eg.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    result = k_core(G)
    assert max(result) <= 2
