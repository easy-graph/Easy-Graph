from easygraph.utils import *
import easygraph as eg

__all__ = [
    "k_core",
]


from typing import TYPE_CHECKING
from typing import List
from typing import Union


if TYPE_CHECKING:
    from easygraph import Graph


@hybrid("cpp_k_core")
def k_core(G: "Graph", k: int = None, return_graph: bool = False) -> Union["Graph", List]:
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
    return list(core.values())


    # Initialize a dictionary to store the degrees of the nodes
    # degrees = dict(H.degree())
    # k = min(degrees.values())

    # Repeat until all nodes have degree < k
    # while True:
    #     # Find the nodes with degree < k
    #     to_remove = [n for n in H.nodes if degrees[n] <= k]

    #     # If there are no such nodes, we're done
    #     if not to_remove:
    #         break

    #     # Remove the nodes and their incident edges
    #     for n in to_remove:
    #         neighbors = list(H.neighbors(n))  # type: ignore
    #         H.remove_node(n)

    #         # Update the degrees of the remaining nodes
    #         for neighbor in neighbors:
    #             if neighbor in degrees:
    #                 degrees[neighbor] -= 1
    
    # # check if the minimum degree is now less than k
    # if min(degrees.values()) > k:
    #     # k = min(degrees.values())
    #     core = k_core(H, min(degrees.values()))
    # else:
    #     core = [degrees[node] for node in G.nodes]

    # return core
    # res = []
    # for node in G.nodes:
    #     res.append(degrees[node])

    # if return_graph:
    #     return H
    # else:
    #     return res


# def main():
#     G = eg.Graph()
#     G.add_edges_from([(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4),(5,6)])
#     res = k_core(G)
#     print(res)

# if __name__ == "__main__":
#     main()