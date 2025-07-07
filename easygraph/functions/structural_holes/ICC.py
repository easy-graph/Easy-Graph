import easygraph as eg

from easygraph.utils import *


__all__ = ["ICC", "BICC", "AP_BICC"]


def inverse_closeness_centrality(G, v):
    if len(G) <= 1:
        return 0
    c_v = sum(eg.Dijkstra(G, v).values()) / (len(G) - 1)
    return c_v


def bounded_inverse_closeness_centrality(G, v, l):
    if len(G) <= 1:
        return 0
    queue = []
    queue.append(v)
    seen = set()
    seen.add(v)
    shortest_path = eg.Floyd(G)
    result = 0
    while len(queue) > 0:
        vertex = queue.pop(0)
        if shortest_path[v][vertex] == l + 1:
            break
        nodes = G.neighbors(node=vertex)
        for w in nodes:
            if w not in seen:
                queue.append(w)
                seen.add(w)
                result += shortest_path[v][w]
    return result / (len(G) - 1)


def Modified_DFS(G, u, V, time, n):
    V[u]["color"] = "black"
    time += 1
    n -= 1
    V[u]["discovered"] = time
    V[u]["lowest"] = time
    cc0 = n
    V[u]["descendant"] = 0
    root = u
    for edge in G.edges:
        u, v = edge[:2]
        if V[u]["color"] == "white":
            V[u]["color"] = "grey"
            V[v]["parent"] = u
            V[u]["child"] += 1
            V, time, n = Modified_DFS(G, v, V, time, n)
            V[u]["descendant"] = V[u]["descendant"] + V[v]["descendant"]
            V[u]["lowest"] = min(V[u]["lowest"], V[v]["lowest"])
            if V[v]["lowest"] >= V[u]["discovered"] or root == u and V[u]["child"] > 1:
                V[u]["c"] += V[v]["descendant"] * (n - V[v]["descendant"] - 1)
                cc0 -= V[v]["descendant"]
        elif v != V[u]["parent"]:
            V[u]["lowest"] = min(V[u]["lowest"], V[v]["discovered"])
    V[u]["c"] += cc0 * (n - cc0 - 1)
    return V, time, n


def approximate_inverse_closeness_centrality(G):
    V = {}
    for i in G.nodes:
        V[i] = {}
        V[i]["child"] = 0
        V[i]["color"] = "white"
        V[i]["c"] = 0
        V[i]["parent"] = None
        V[i]["discovered"] = 0
        V[i]["lowest"] = 0
        V[i]["descendant"] = 0
    time = 0
    n = len(G)
    for u in G.nodes:
        if V[u]["color"] == "white":
            V, time, n = Modified_DFS(G, u, V, time, n)
    return V


@not_implemented_for("multigraph")
def ICC(G, k):
    """an efficient algorithm for structural hole spanners detection.

    Returns top k nodes as structural hole spanners,
    Algorithm 1 of [1]_

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    k : int
        top - k structural hole spanners

    Returns
    -------
    V : list
        The list of top-k structural hole spanners.

    Examples
    --------
    Returns the top k nodes as structural hole spanners, using **ICC**.

    >>> ICC(G,k=3)

    References
    ----------
    .. [1] https://dl.acm.org/doi/10.1145/2806416.2806431

    """
    Q = []
    V = []
    for v in G.nodes:
        i_c = inverse_closeness_centrality(G, v)
        if len(Q) < k:
            Q.append([v, i_c])
            continue
        MAX = 0
        t = v
        for i in Q:
            if MAX < i[1]:
                MAX = i[1]
                t = i[0]
        if i_c < MAX:
            Q.remove([t, MAX])
            Q.append([v, i_c])
    for i in Q:
        V.append(i[0])
    return V


@not_implemented_for("multigraph")
def BICC(G, k, K, l):
    """an efficient algorithm for structural hole spanners detection.

    Returns top k nodes as structural hole spanners,
    Algorithm 2 of [1]_

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    k : int
        top - k structural hole spanners

    K : int
        the number of candidates K for the top-k hole spanners

    l : int
        level-l neighbors of nodes

    Returns
    -------
    V : list
        The list of top-k structural hole spanners.

    Examples
    --------
    Returns the top k nodes as structural hole spanners, using **BICC**.

    >>> BICC(G,k=3,K=5,l=4)

    References
    ----------
    .. [1] https://dl.acm.org/doi/10.1145/2806416.2806431

    """
    H = []
    V = []
    for v in G.nodes:
        b_i_c = bounded_inverse_closeness_centrality(G, v, l)
        if len(H) < K:
            H.append([v, b_i_c])
            continue
        MIN = 10000000
        t = v
        for i in H:
            if MIN > i[1]:
                MIN = i[1]
                t = i[0]
        if b_i_c > MIN:
            H.remove([t, MIN])
            H.append([v, b_i_c])
    for i in H:
        v = i[0]
        i_c = inverse_closeness_centrality(G, v)
        if len(V) < k:
            V.append([v, i_c])
            continue
        MAX = 0
        t = v
        for i in V:
            if MAX < i[1]:
                MAX = i[1]
                t = i[0]
        if i_c < MAX:
            V.remove([t, MAX])
            V.append([v, i_c])
    VS = []
    for i in V:
        VS.append(i[0])
    return VS


@not_implemented_for("multigraph")
def AP_BICC(G, k, K, l):
    """an efficient algorithm for structural hole spanners detection.

    Returns top k nodes as structural hole spanners,
    Algorithm 3 of [1]_

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    k : int
        top - k structural hole spanners

    K : int
        the number of candidates K for the top-k hole spanners

    l : int
        level-l neighbors of nodes

    Returns
    -------
    V : list
        The list of top-k structural hole spanners.

    Examples
    --------
    Returns the top k nodes as structural hole spanners, using **AP_BICC**.

    >>> AP_BICC(G,k=3,K=5,l=4)

    References
    ----------
    .. [1] https://dl.acm.org/doi/10.1145/2806416.2806431

    """
    V = []
    T = []
    A = {}
    A = approximate_inverse_closeness_centrality(G)
    for v in A:
        if len(T) < k:
            T.append([v, A[v]["c"]])
            continue
        MIN = 10000000
        t = v
        for i in T:
            if MIN > i[1]:
                MIN = i[1]
                t = i[0]
        if A[v]["c"] > MIN:
            T.remove([t, MIN])
            T.append([v, A[v]["c"]])
    if len(T) < k:
        U = {}
        for i in G.nodes:
            if i not in A:
                U.append(i)
        kk = k - len(T)
        Q = []
        for v in U:
            b_i_c = bounded_inverse_closeness_centrality(G, v, l)
            if len(Q) < K:
                Q.append([v, b_i_c])
            else:
                MIN = 10000000
                t = v
                for i in Q:
                    if MIN > i[1]:
                        MIN = i[1]
                        t = i[0]
                if b_i_c > MIN:
                    Q.remove([t, MIN])
                    Q.append([v, b_i_c])
    while len(T) != k:
        MAX = 0
        t = None
        for i in Q:
            if MAX < i[1]:
                MAX = i[1]
                t = i[0]
        T.append([t, A[t]["c"]])
    for i in T:
        V.append(i[0])
    return V


if __name__ == "__main__":
    G = eg.datasets.get_graph_karateclub()
    print(ICC(G, 3))
    print(BICC(G, 3, 5, 3))
    print(AP_BICC(G, 3, 5, 3))
