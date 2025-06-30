import easygraph as eg
import numpy as np

from easygraph.utils import *


__all__ = ["NOBE", "NOBE_GA"]


@not_implemented_for("multigraph")
def NOBE(G, K):
    """Graph embedding via NOBE[1].

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    K : int
        Embedding dimension k

    Returns
    -------
    Y : list
        list of embedding vectors (y1, y2, · · · , yn)

    Examples
    --------
    >>> NOBE(G,K=15)

    References
    ----------
    .. [1] https://www.researchgate.net/publication/325004496_On_Spectral_Graph_Embedding_A_Non-Backtracking_Perspective_and_Graph_Approximation

    """
    dict = {}
    a = 0
    for i in G.nodes:
        dict[i] = a
        a += 1
    LG = graph_to_d_atleast2(G)
    N = len(G)
    P, pair = Transition(LG)
    V = eigs_nodes(P, K)
    Y = embedding(V, pair, K, N, dict, G)
    return Y


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
def NOBE_GA(G, K):
    """Graph embedding via NOBE-GA[1].

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    K : int
        Embedding dimension k

    Returns
    -------
    Y : list
        list of embedding vectors (y1, y2, · · · , yn)

    Examples
    --------
    >>> NOBE_GA(G,K=15)

    References
    ----------
    .. [1] https://www.researchgate.net/publication/325004496_On_Spectral_Graph_Embedding_A_Non-Backtracking_Perspective_and_Graph_Approximation

    """
    from scipy.sparse.linalg import eigs

    N = len(G)
    A = np.eye(N, N)
    for i in G.edges:
        (u, v, t) = i
        u = int(u) - 1
        v = int(v) - 1
        A[u, v] = 1
    degree = G.degree()
    D_inv = np.zeros([N, N])
    a = 0
    for i in degree:
        D_inv[a, a] = 1 / degree[i]
        a += 1
    D_I_inv = np.zeros([N, N])
    b = 0
    for i in degree:
        if degree[i] > 1:
            D_I_inv[b, b] = 1 / (degree[i] - 1)
        b += 1
    I = np.identity(N)
    M_D = 0.5 * A * D_I_inv * (I - D_inv)
    D_D = 0.5 * I
    T_ua = np.zeros([2 * N, 2 * N])
    T_ua[0:N, 0:N] = M_D
    T_ua[N : 2 * N, N : 2 * N] = M_D
    T_ua[N : 2 * N, 0:N] = D_D
    T_ua[0:N, N : 2 * N] = D_D
    Y1, Y = eigs(T_ua, K + 1, which="LR")
    Y = Y[0:N, :-1]
    return Y


def graph_to_d_atleast2(G):
    n = len(G)
    LG = eg.Graph()
    LG = G.copy()
    new_node = n
    degree = LG.degree()
    node = LG.nodes.copy()
    for i in node:
        if degree[i] == 1:
            for neighbors in LG.neighbors(node=i):
                LG.add_edge(i, new_node)
                LG.add_edge(new_node, neighbors)
                break
            new_node = new_node + 1
    return LG


def Transition(LG):
    N = len(LG)
    M = LG.size()
    LLG = eg.DiGraph()
    for i in LG.edges:
        (u, v, t) = i
        LLG.add_edge(u, v)
        LLG.add_edge(v, u)
    degree = LLG.degree()
    P = np.zeros([2 * M, 2 * M])
    pair = []
    k = 0
    l = 0
    for i in LLG.edges:
        l = 0
        for j in LLG.edges:
            (u, v, t) = i
            (x, y, z) = j
            if v == x and u != y:
                P[k][l] = 1 / (degree[v] - 1)
            l += 1
        k += 1
    a = 0
    for i in LLG.edges:
        (u, v, t) = i
        pair.append([u, v])
        a += 1
    return P, pair


def eigs_nodes(P, K):
    from scipy.sparse.linalg import eigs

    M = np.size(P, 0)
    L = np.zeros([M, M])
    I = np.identity(M)
    P_T = P.T
    L = I - (P + P_T) / 2
    U, D = eigs(L, K + 1, which="LR")
    D = D[:, :-1]
    V = np.zeros([M, K], dtype=complex)
    a = 0
    for i in D:
        V[a] = i
        a += 1
    return V


def embedding(V, pair, K, N, dict, G):
    Y = np.zeros([N, K], dtype=complex)
    idx = 0
    for i in pair:
        [v, u] = i
        if u in G.nodes:
            t = dict[u]
            for j in range(0, len(V[idx])):
                Y[t, j] += V[idx, j]
            idx += 1
    return Y
