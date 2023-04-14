from easygraph.utils import *


__all__ = [
    "k_core",
]


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from easygraph import Graph


@hybrid("cpp_k_core")
def k_core(G: "Graph", k: int = 1):
    # Create a shallow copy of the input graph
    H = G.copy()

    # Initialize a dictionary to store the degrees of the nodes
    degrees = dict(H.degree())

    # Repeat until all nodes have degree < k
    while True:
        # Find the nodes with degree < k
        to_remove = [n for n in H.nodes if degrees[n] < k]

        # If there are no such nodes, we're done
        if not to_remove:
            break

        # Remove the nodes and their incident edges
        H.remove_nodes_from(to_remove)

        # Update the degrees of the remaining nodes
        degrees = dict(H.degree())

    return H


def test_k_core():
    from easygraph import Graph

    G = Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])
    H = k_core(G, k=2)
    assert sorted(H.nodes.keys()) == sorted([1, 2, 3, 4])
    assert sorted((x, y) for x, y, _ in H.edges) == sorted(
        [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    )
