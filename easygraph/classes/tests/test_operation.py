import easygraph as eg
import pytest

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
