import math

from itertools import combinations
from typing import List

from easygraph.utils import *


__all__ = ["get_structural_holes_HIS"]


@not_implemented_for("multigraph")
def get_structural_holes_HIS(G, C: List[frozenset], epsilon=1e-4, weight="weight"):
    """Structural hole spanners detection via HIS method.

    Both **HIS** and **MaxD** are methods in [1]_.
    The authors developed these two methods to find the structural holes spanners,
    based on theory of information diffusion.

    Returns the value of `S`, `I`, `H` ,defined in **HIS** of [1], of each node in the graph.
    Note that `H` quantifies the possibility that a node is a structural hole spanner.
    To use `HIS` method, you should provide the community detection result as parameter.

    Parameters
    ----------
    C : list of frozenset
        Each frozenset denotes a community of nodes.

    epsilon : float
        The threshold value.

    weight : string, optional (default : 'weight')
        The key for edge weight.

    Returns
    -------
    S : list of tuple
        The `S` value in [1]_.

    I : float
        The `I` value in [1]_.

    H : float
        The `H` value in [1]_.

    See Also
    --------
    MaxD

    Examples
    --------

    >>> get_structural_holes_HIS(G,
    ...                          C = [frozenset([1,2,3]), frozenset([4,5,6])], # Two communities
    ...                          epsilon = 0.01,
    ...                          weight = 'weight'
    ...                          )


    References
    ----------
    .. [1] https://www.aminer.cn/structural-hole

    """
    # S: List[subset_index]
    S = []
    for community_subset_size in range(2, len(C) + 1):
        S.extend(list(combinations(range(len(C)), community_subset_size)))
    # I: dict[node][cmnt_index]
    # H: dict[node][subset_index]

    if not G.nodes or not C:
        return [], {}, {}

    I, H = initialize(G, C, S, weight=weight)

    if not S:
        return S, I, H

    alphas = [0.3 for i in range(len(C))]  # list[cmnt_index]
    betas = [(0.5 - math.pow(0.5, len(subset))) for subset in S]  # list[subset_index]

    while True:
        P = update_P(G, C, alphas, betas, S, I, H)  # dict[node][cmnt_index]
        I_new, H_new = update_I_H(G, C, S, P, I)
        if is_convergence(G, C, I, I_new, epsilon):
            break
        else:
            I, H = I_new, H_new
    return S, I, H


def initialize(G, C: List[frozenset], S: [tuple], weight="weight"):
    I, H = dict(), dict()
    for node in G.nodes:
        I[node] = dict()
        H[node] = dict()

    for node in G.nodes:
        for index, community in enumerate(C):
            if node in community:
                # TODO: add PageRank or HITS to initialize I
                I[node][index] = G.degree(weight=weight)[node]
            else:
                I[node][index] = 0

    for node in G.nodes:
        for index, subset in enumerate(S):
            H[node][index] = min(I[node][i] for i in subset)

    return I, H


def update_P(G, C, alphas, betas, S, I, H):
    P = dict()
    for node in G.nodes:
        P[node] = dict()

    for node in G.nodes:
        for cmnt_index in range(len(C)):
            subsets_including_current_cmnt = []
            for subset_index in range(len(S)):
                if cmnt_index in S[subset_index]:
                    subsets_including_current_cmnt.append(
                        alphas[cmnt_index] * I[node][cmnt_index]
                        + betas[subset_index] * H[node][subset_index]
                    )
            P[node][cmnt_index] = max(subsets_including_current_cmnt)
    return P


def update_I_H(G, C, S, P, I):
    I_new, H_new = dict(), dict()
    for node in G.nodes:
        I_new[node] = dict()
        H_new[node] = dict()

    for node in G.nodes:
        for cmnt_index in range(len(C)):
            P_max = max(P[neighbour][cmnt_index] for neighbour in G.adj[node])
            I_new[node][cmnt_index] = (
                P_max if (P_max > I[node][cmnt_index]) else I[node][cmnt_index]
            )
        for subset_index, subset in enumerate(S):
            H_new[node][subset_index] = min(I_new[node][i] for i in subset)
    return I_new, H_new


def is_convergence(G, C, I, I_new, epsilon):
    deltas = []
    for node in G.nodes:
        for cmnt_index in range(len(C)):
            deltas.append(abs(I[node][cmnt_index] - I_new[node][cmnt_index]))
    return max(deltas) < epsilon
