import easygraph as eg
import pytest

from easygraph.functions.basic import average_degree


def test_average_degree_basic():
    G = eg.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    assert average_degree(G) == pytest.approx(4 / 3)


def test_average_degree_empty_graph():
    G = eg.Graph()
    with pytest.raises(ZeroDivisionError):
        average_degree(G)


def test_average_degree_self_loop():
    G = eg.Graph()
    G.add_edge(1, 1)  # self-loop
    # Self-loop counts as 2 towards degree of node 1
    assert average_degree(G) == pytest.approx(2.0)


def test_average_degree_with_isolated_node():
    G = eg.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    G.add_node(4)  # isolated node
    assert average_degree(G) == pytest.approx(1.0)


def test_average_degree_directed_graph():
    G = eg.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    assert average_degree(G) == pytest.approx(2.0)


def test_average_degree_invalid_input():
    with pytest.raises(AttributeError):
        average_degree(None)
