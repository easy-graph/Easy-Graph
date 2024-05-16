"""Unit tests for the :mod:`easygraph.functions.isolates` module."""

import easygraph as eg


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
