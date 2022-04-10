import easygraph as eg
import random
from easygraph.utils import *

__all__ = ["enumerate_subgraph"]


@not_implemented_for("multigraph")
def enumerate_subgraph(G, k: int):
    """
    Returns the motifs.
    Motifs are small weakly connected induced subgraphs of a given structure in a graph.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph.

    k : int
        The size of the motifs to search for.

    Returns
    ----------
    k_subgraphs : list
        The motifs.

    References
    ----------
    .. [1] Wernicke, Sebastian. "Efficient detection of network motifs." 
        IEEE/ACM transactions on computational biology and bioinformatics 3.4 (2006): 347-359.

    """
    k_subgraphs = []
    for v, _ in G.nodes.items():
        Vextension = set([u for u in G.adj[v] if u > v])
        extend_subgraph(G, set([v]), Vextension, v, k, k_subgraphs)
    return k_subgraphs


def extend_subgraph(G, Vsubgraph: set, Vextension: set, v: int, k: int,
                    k_subgraphs: list):
    if len(Vsubgraph) == k:
        k_subgraphs.append(Vsubgraph)
        return
    while len(Vextension) > 0:
        w = random.choice(tuple(Vextension))
        Vextension.remove(w)
        NexclwVsubgraph = exclusive_neighborhood(G, w, Vsubgraph)
        VpExtension = Vextension | set([u for u in NexclwVsubgraph if u > v])
        extend_subgraph(G, Vsubgraph | set([w]), VpExtension, v, k,
                        k_subgraphs)


def exclusive_neighborhood(G, v: int, vp: set):
    Nv = set(G.adj[v])
    NVp = set([u for n in vp for u in G.adj[n]]) | vp
    return Nv - NVp
