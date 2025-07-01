import random

import easygraph as eg

from easygraph.utils import *


__all__ = ["enumerate_subgraph", "random_enumerate_subgraph"]


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
        Vextension = {u for u in G.adj[v] if u > v}
        extend_subgraph(G, {v}, Vextension, v, k, k_subgraphs)
    return k_subgraphs


def extend_subgraph(
    G, Vsubgraph: set, Vextension: set, v: int, k: int, k_subgraphs: list
):
    if len(Vsubgraph) == k:
        k_subgraphs.append(Vsubgraph)
        return
    while len(Vextension) > 0:
        w = random.choice(tuple(Vextension))
        Vextension.remove(w)
        NexclwVsubgraph = exclusive_neighborhood(G, w, Vsubgraph)
        VpExtension = Vextension | {u for u in NexclwVsubgraph if u > v}
        extend_subgraph(G, Vsubgraph | {w}, VpExtension, v, k, k_subgraphs)


def exclusive_neighborhood(G, v: int, vp: set):
    Nv = set(G.adj[v])
    NVp = {u for n in vp for u in G.adj[n]} | vp
    return Nv - NVp


@not_implemented_for("multigraph")
def random_enumerate_subgraph(G, k: int, cut_prob: list):
    """
    Returns the motifs.
    Motifs are small weakly connected induced subgraphs of a given structure in a graph.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph.

    k : int
        The size of the motifs to search for.

    cut_prob : list
        list of probabilities for cutting the search tree at a given level.

    Returns
    ----------
    k_subgraphs : list
        The motifs.

    References
    ----------
    .. [1] Wernicke, Sebastian. "A faster algorithm for detecting network motifs."
    International Workshop on Algorithms in Bioinformatics. Springer, Berlin, Heidelberg, 2005.

    """
    if len(cut_prob) != k:
        raise eg.EasyGraphError("length of cut_prob invalid, should equal to k")

    k_subgraphs = []
    for v, _ in G.nodes.items():
        if random.random() > cut_prob[0]:
            continue
        Vextension = {u for u in G.adj[v] if u > v}
        random_extend_subgraph(G, {v}, Vextension, v, k, k_subgraphs, cut_prob)
    return k_subgraphs


def random_extend_subgraph(
    G,
    Vsubgraph: set,
    Vextension: set,
    v: int,
    k: int,
    k_subgraphs: list,
    cut_prob: list,
):
    if len(Vsubgraph) == k:
        k_subgraphs.append(Vsubgraph)
        return
    while len(Vextension) > 0:
        w = random.choice(tuple(Vextension))
        Vextension.remove(w)
        NexclwVsubgraph = exclusive_neighborhood(G, w, Vsubgraph)
        VpExtension = Vextension | {u for u in NexclwVsubgraph if u > v}
        if random.random() > cut_prob[len(Vsubgraph)]:
            continue
        random_extend_subgraph(
            G, Vsubgraph | {w}, VpExtension, v, k, k_subgraphs, cut_prob
        )
