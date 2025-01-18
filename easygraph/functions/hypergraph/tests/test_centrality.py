import easygraph as eg
import pytest


@pytest.fixture()
def g1():
    e_list = [(0, 1, 2, 5), (0, 1), (2, 3, 4), (1, 2, 4)]
    g = eg.Hypergraph(6, e_list=e_list)
    return g


@pytest.fixture()
def g2():
    e_list = [(1, 2, 3), (0, 1, 3), (0, 1), (2, 4, 3), (2, 3)]
    e_weight = [0.5, 1, 0.5, 1, 0.5]
    g = eg.Hypergraph(5, e_list=e_list, e_weight=e_weight)
    return g


def test_degree_centrality(g1, g2):
    print(eg.hyepergraph_degree_centrality(g1))
    print(eg.hyepergraph_degree_centrality(g2))
    assert eg.hyepergraph_degree_centrality(g1) == {0: 2, 1: 3, 2: 3, 3: 1, 4: 2, 5: 1}
    assert eg.hyepergraph_degree_centrality(g2) == {0: 2, 1: 3, 2: 3, 3: 4, 4: 1}
