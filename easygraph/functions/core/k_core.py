from easygraph.utils import *


__all__ = [
    "k_core",
]


from typing import TYPE_CHECKING
from typing import List
from typing import Union


if TYPE_CHECKING:
    from easygraph import Graph


@hybrid("cpp_k_core")
def k_core(G: "Graph", k: int = 1, return_graph: bool = False) -> Union["Graph", List]:
    """
    Returns the k-core of G.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    Parameters
    ----------
    G : EasyGraph graph
      A graph or directed graph
    k : int, optional
      The order of the core.  If not specified return the main core.
    return_graph : bool, optional
        If True, return the k-core as a graph.  If False, return a list of nodes.

    Returns
    -------
    G : EasyGraph graph, if return_graph is True, else a list of nodes
      The k-core subgraph
    """
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
            neighbors = list(H.neighbors(n))  # type: ignore
            H.remove_node(n)

            # Update the degrees of the remaining nodes
            for neighbor in neighbors:
                if neighbor in degrees:
                    degrees[neighbor] -= 1

    if return_graph:
        return H
    else:
        return list(H.nodes.keys())
