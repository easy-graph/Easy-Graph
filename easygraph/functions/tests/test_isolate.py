"""Unit tests for the :mod:`easygraph.functions.isolates` module."""

import easygraph as eg
import pytest


def test_is_isolate():
    G = eg.Graph()
    G.add_edge(0, 1)
    G.add_node(2)
    assert not eg.is_isolate(G, 0)
    assert not eg.is_isolate(G, 1)
    assert eg.is_isolate(G, 2)


def test_isolates():
    G = eg.Graph()
    G.add_edge(0, 1)
    G.add_nodes_from([2, 3])
    assert sorted(eg.isolates(G)) == [2, 3]


def test_number_of_isolates():
    G = eg.Graph()
    G.add_edge(0, 1)
    G.add_nodes_from([2, 3])
    assert eg.number_of_isolates(G) == 2


def test_empty_graph_isolates():
    G = eg.Graph()
    assert list(eg.isolates(G)) == []
    assert eg.number_of_isolates(G) == 0


def test_all_isolates_graph():
    G = eg.Graph()
    G.add_nodes_from(range(5))
    assert sorted(eg.isolates(G)) == list(range(5))
    assert all(eg.is_isolate(G, n) for n in G.nodes)
    assert eg.number_of_isolates(G) == 5


def test_directed_graph_sources_and_sinks_not_isolates():
    G = eg.DiGraph()
    G.add_edges_from([(1, 2), (2, 3)])
    G.add_node(4)  # truly isolated
    assert eg.is_isolate(G, 4)
    assert not eg.is_isolate(G, 1)  # has out-degree
    assert not eg.is_isolate(G, 3)  # has in-degree
    assert sorted(eg.isolates(G)) == [4]
    assert eg.number_of_isolates(G) == 1


def test_selfloop_not_isolate():
    G = eg.Graph()
    G.add_node(1)
    G.add_edge(1, 1)
    assert not eg.is_isolate(G, 1)
    assert list(eg.isolates(G)) == []
    assert eg.number_of_isolates(G) == 0


def test_weighted_edges_isolate_behavior():
    G = eg.Graph()
    G.add_edge(1, 2, weight=5)
    G.add_node(3)
    assert eg.is_isolate(G, 3)
    assert not eg.is_isolate(G, 1)
    assert eg.number_of_isolates(G) == 1


def test_remove_isolate_then_check():
    G = eg.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edge(1, 2)
    assert 3 in eg.isolates(G)
    G.remove_node(3)
    assert 3 not in G
    assert 3 not in list(eg.isolates(G))


def test_mixed_isolates_and_edges():
    G = eg.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([(0, 1), (1, 2)])
    # 3 and 4 are isolates
    assert set(eg.isolates(G)) == {3, 4}
    assert eg.number_of_isolates(G) == 2
    for node in [0, 1, 2]:
        assert not eg.is_isolate(G, node)
    for node in [3, 4]:
        assert eg.is_isolate(G, node)
