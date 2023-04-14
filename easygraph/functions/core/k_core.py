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
        for n in to_remove:
            neighbors = list(H.neighbors(n))
            H.remove_node(n)

            # Update the degrees of the remaining nodes
            for neighbor in neighbors:
                if neighbor in degrees:
                    degrees[neighbor] -= 1

    return H
