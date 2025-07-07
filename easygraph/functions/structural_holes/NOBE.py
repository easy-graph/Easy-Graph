import easygraph as eg
import numpy as np

from easygraph.utils import *


__all__ = ["NOBE_SH", "NOBE_GA_SH"]


@not_implemented_for("multigraph")
def NOBE_SH(G, K, topk):
    """detect SH spanners via NOBE[1].

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    K : int
        Embedding dimension k

    topk : int
        top - k structural hole spanners

    Returns
    -------
    SHS : list
        The top-k structural hole spanners.

    Examples
    --------
    >>> NOBE_SH(G,K=8,topk=5)

    References
    ----------
    .. [1] https://www.researchgate.net/publication/325004496_On_Spectral_Graph_Embedding_A_Non-Backtracking_Perspective_and_Graph_Approximation

    """
    if K <= 0:
        raise ValueError("Embedding dimension K must be a positive integer.")
    if topk <= 0:
        raise ValueError("Parameter topk must be a positive integer.")
    from sklearn.cluster import KMeans

    Y = eg.graph_embedding.NOBE(G, K)
    dict = {}
    a = 0
    for i in G.nodes:
        dict[i] = a
        a += 1
    if isinstance(Y[0, 0], complex):
        Y = abs(Y)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(Y)
    com = {}
    cluster = {}
    for i in dict:
        com[i] = kmeans.labels_[dict[i]]
    for i in com:
        if com[i] in cluster:
            cluster[com[i]].append(i)
        else:
            cluster[com[i]] = []
            cluster[com[i]].append(i)
    vector = {}
    for i in dict:
        vector[i] = Y[dict[i]]
    rds = RDS(com, cluster, vector, K)
    rds_sort = sorted(rds.items(), key=lambda d: d[1], reverse=True)
    SHS = list()
    a = 0
    for i in rds_sort:
        SHS.append(i[0])
        a += 1
        if a == topk:
            break
    return SHS


@not_implemented_for("multigraph")
def NOBE_GA_SH(G, K, topk):
    """detect SH spanners via NOBE-GA[1].

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    K : int
        Embedding dimension k

    topk : int
        top - k structural hole spanners

    Returns
    -------
    SHS : list
        The top-k structural hole spanners.

    Examples
    --------
    >>> NOBE_GA_SH(G,K=8,topk=5)

    References
    ----------
    .. [1] https://www.researchgate.net/publication/325004496_On_Spectral_Graph_Embedding_A_Non-Backtracking_Perspective_and_Graph_Approximation

    """
    if K <= 0:
        raise ValueError("Embedding dimension K must be a positive integer.")
    if topk <= 0:
        raise ValueError("Parameter topk must be a positive integer.")
    from sklearn.cluster import KMeans

    Y = eg.NOBE_GA(G, K)
    if isinstance(Y[0, 0], complex):
        Y = abs(Y)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(Y)
    com = {}
    cluster = {}
    a = 0
    for i in G.nodes:
        com[i] = kmeans.labels_[a]
        a += 1
    for i in com:
        if com[i] in cluster:
            cluster[com[i]].append(i)
        else:
            cluster[com[i]] = []
            cluster[com[i]].append(i)
    vector = {}
    a = 0
    for i in G.nodes:
        vector[i] = Y[a]
        a += 1
    rds = RDS(com, cluster, vector, K)
    rds_sort = sorted(rds.items(), key=lambda d: d[1], reverse=True)
    SHS = list()
    a = 0
    for i in rds_sort:
        SHS.append(i[0])
        a += 1
        if a == topk:
            break
    return SHS


def RDS(com, cluster, vector, K):
    rds = {}
    Uc = {}
    Rc = {}
    for i in cluster:
        sum_vec = np.zeros(K)
        for j in cluster[i]:
            sum_vec += vector[j]
        Uc[i] = sum_vec / len(cluster[i])
    for i in cluster:
        sum_dist = 0
        for j in cluster[i]:
            sum_dist += np.linalg.norm(vector[j] - Uc[i])
        Rc[i] = sum_dist
    for i in com:
        maxx = 0
        fenzi = np.linalg.norm(vector[i] - Uc[com[i]]) / Rc[com[i]]
        for j in cluster:
            fenmu = np.linalg.norm(vector[i] - Uc[j]) / Rc[j]
            if maxx < fenzi / fenmu:
                maxx = fenzi / fenmu
        rds[i] = maxx
    return rds


if __name__ == "__main__":
    G = eg.datasets.get_graph_karateclub()
    print(NOBE_SH(G, K=2, topk=3))
    print(NOBE_GA_SH(G, K=2, topk=3))
