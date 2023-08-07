import pytest


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

    # Create EasyGraph and NetworkX graphs from the edge list
    G = Graph()
    G_nx = nx.Graph()
    G.add_edges_from(edges)
    G_nx.add_edges_from(edges)

    # Compute the k-core of the graphs using the k_core function and nx.k_core
    H = k_core(G)
    H_nx = nx.core_number(G_nx)  # type: ignore
    assert H == list(H_nx.values())
