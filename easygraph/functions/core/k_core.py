import easygraph as eg

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
def k_core(G: "Graph") -> Union["Graph", List]:
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
    degrees = dict(G.degree())
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: list(G.neighbors(v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    ret = [0.0 for i in range(len(G))]
    for i in range(len(ret)):
        ret[i] = core[G.index2node[i]]
    return ret
