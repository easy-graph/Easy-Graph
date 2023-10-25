import math
import random

import easygraph as eg

from easygraph.classes.graph import Graph


__all__ = [
    "erdos_renyi_M",
    "erdos_renyi_P",
    "fast_erdos_renyi_P",
    "WS_Random",
    "graph_Gnm",
]


def erdos_renyi_M(n, edge, directed=False, FilePath=None):
    """Given the number of nodes and the number of edges, return an Erdős-Rényi random graph, and store the graph in a document.

    Parameters
    ----------
    n : int
        The number of nodes.
    edge : int
        The number of edges.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.
    FilePath : string
        The file for storing the output graph G.

    Returns
    -------
    G : graph
        an Erdős-Rényi random graph.

    Examples
    --------
    Returns an Erdős-Rényi random graph G.

    >>> erdos_renyi_M(100,180,directed=False,FilePath="/users/fudanmsn/downloads/RandomNetwork.txt")

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    if directed:
        G = eg.DiGraph()
        adjacent = {}
        mmax = n * (n - 1)
        if edge >= mmax:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        G.add_edge(i, j)
                        if i not in adjacent:
                            adjacent[i] = []
                            adjacent[i].append(j)
                        else:
                            adjacent[i].append(j)
            return G
        count = 0
        while count < edge:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or G.has_edge(i, j):
                continue
            else:
                count = count + 1
                if i not in adjacent:
                    adjacent[i] = []
                    adjacent[i].append(j)
                else:
                    adjacent[i].append(j)
                G.add_edge(i, j)
    else:
        G = eg.Graph()
        adjacent = {}
        mmax = n * (n - 1) / 2
        if edge >= mmax:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        G.add_edge(i, j)
                        if i not in adjacent:
                            adjacent[i] = []
                            adjacent[i].append(j)
                        else:
                            adjacent[i].append(j)
                        if j not in adjacent:
                            adjacent[j] = []
                            adjacent[j].append(i)
                        else:
                            adjacent[j].append(i)
            return G
        count = 0
        while count < edge:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or G.has_edge(i, j):
                continue
            else:
                count = count + 1
                if i not in adjacent:
                    adjacent[i] = []
                    adjacent[i].append(j)
                else:
                    adjacent[i].append(j)
                if j not in adjacent:
                    adjacent[j] = []
                    adjacent[j].append(i)
                else:
                    adjacent[j].append(i)
                G.add_edge(i, j)

    writeRandomNetworkToFile(n, adjacent, FilePath)
    return G


def erdos_renyi_P(n, p, directed=False, FilePath=None):
    """Given the number of nodes and the probability of edge creation, return an Erdős-Rényi random graph, and store the graph in a document.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.
    FilePath : string
        The file for storing the output graph G.

    Returns
    -------
    G : graph
        an Erdős-Rényi random graph.

    Examples
    --------
    Returns an Erdős-Rényi random graph G

    >>> erdos_renyi_P(100,0.5,directed=False,FilePath="/users/fudanmsn/downloads/RandomNetwork.txt")

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    if directed:
        G = eg.DiGraph()
        adjacent = {}
        probability = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                probability = random.random()
                if probability < p:
                    if i not in adjacent:
                        adjacent[i] = []
                        adjacent[i].append(j)
                    else:
                        adjacent[i].append(j)
                    G.add_edge(i, j)
    else:
        G = eg.Graph()
        adjacent = {}
        probability = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                probability = random.random()
                if probability < p:
                    if i not in adjacent:
                        adjacent[i] = []
                        adjacent[i].append(j)
                    else:
                        adjacent[i].append(j)
                    if j not in adjacent:
                        adjacent[j] = []
                        adjacent[j].append(i)
                    else:
                        adjacent[j].append(i)
                    G.add_edge(i, j)

    writeRandomNetworkToFile(n, adjacent, FilePath)
    return G


def fast_erdos_renyi_P(n, p, directed=False, FilePath=None):
    """Given the number of nodes and the probability of edge creation, return an Erdős-Rényi random graph, and store the graph in a document. Use this function for generating a huge scale graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.
    FilePath : string
        The file for storing the output graph G.

    Returns
    -------
    G : graph
        an Erdős-Rényi random graph.

    Examples
    --------
    Returns an Erdős-Rényi random graph G

    >>> erdos_renyi_P(100,0.5,directed=False,FilePath="/users/fudanmsn/downloads/RandomNetwork.txt")

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    if directed:
        G = eg.DiGraph()
        w = -1
        lp = math.log(1.0 - p)
        v = 0
        adjacent = {}
        while v < n:
            lr = math.log(1.0 - random.random())
            w = w + 1 + int(lr / lp)
            if v == w:  # avoid self loops
                w = w + 1
            while v < n <= w:
                w = w - n
                v = v + 1
                if v == w:  # avoid self loops
                    w = w + 1
            if v < n:
                G.add_edge(v, w)
                if v not in adjacent:
                    adjacent[v] = []
                    adjacent[v].append(w)
                else:
                    adjacent[v].append(w)
    else:
        G = eg.Graph()
        w = -1
        lp = math.log(1.0 - p)
        v = 1
        adjacent = {}
        while v < n:
            lr = math.log(1.0 - random.random())
            w = w + 1 + int(lr / lp)
            while w >= v and v < n:
                w = w - v
                v = v + 1
            if v < n:
                G.add_edge(v, w)
                if v not in adjacent:
                    adjacent[v] = []
                    adjacent[v].append(w)
                else:
                    adjacent[v].append(w)
                if w not in adjacent:
                    adjacent[w] = []
                    adjacent[w].append(v)
                else:
                    adjacent[w].append(v)

    writeRandomNetworkToFile(n, adjacent, FilePath)
    return G


def WS_Random(n, k, p, FilePath=None):
    """Returns a small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    FilePath : string
        The file for storing the output graph G

    Returns
    -------
    G : graph
        a small-world graph

    Examples
    --------
    Returns a small-world graph G

    >>> WS_Random(100,10,0.3,"/users/fudanmsn/downloads/RandomNetwork.txt")

    """
    if k >= n:
        print("k>=n, choose smaller k or larger n")
        return
    adjacent = {}
    G = eg.Graph()
    NUM1 = n
    NUM2 = NUM1 - 1
    K = k
    K1 = K + 1
    N = list(range(NUM1))
    G.add_nodes(N)

    for i in range(NUM1):
        for j in range(1, K1):
            K_add = NUM1 - K
            i_add_j = i + j + 1
            if i >= K_add and i_add_j > NUM1:
                i_add = i + j - NUM1
                G.add_edge(i, i_add)
            else:
                i_add = i + j
                G.add_edge(i, i_add)
            if i not in adjacent:
                adjacent[i] = []
                adjacent[i].append(i_add)
            else:
                adjacent[i].append(i_add)
            if i_add not in adjacent:
                adjacent[i_add] = []
                adjacent[i_add].append(i)
            else:
                adjacent[i_add].append(i)
    for i in range(NUM1):
        for e_del in range(i + 1, i + K1):
            if e_del >= NUM1:
                e_del = e_del - NUM1
            P_random = random.random()
            if P_random < p:
                G.remove_edge(i, e_del)
                adjacent[i].remove(e_del)
                if adjacent[i] == []:
                    adjacent.pop(i)
                adjacent[e_del].remove(i)
                if adjacent[e_del] == []:
                    adjacent.pop(e_del)
                e_add = random.randint(0, NUM2)
                while e_add == i or G.has_edge(i, e_add) == True:
                    e_add = random.randint(0, NUM2)
                G.add_edge(i, e_add)
                if i not in adjacent:
                    adjacent[i] = []
                    adjacent[i].append(e_add)
                else:
                    adjacent[i].append(e_add)
                if e_add not in adjacent:
                    adjacent[e_add] = []
                    adjacent[e_add].append(i)
                else:
                    adjacent[e_add].append(i)
    writeRandomNetworkToFile(n, adjacent, FilePath)
    return G


def writeRandomNetworkToFile(n, adjacent, FilePath):
    if FilePath != None:
        f = open(FilePath, "w+")
    else:
        f = open("RandomNetwork.txt", "w+")
    adjacent = sorted(adjacent.items(), key=lambda d: d[0])
    for i in adjacent:
        i[1].sort()
        for j in i[1]:
            f.write(str(i[0]))
            f.write(" ")
            f.write(str(j))
            f.write("\n")
    f.close()


def graph_Gnm(num_v: int, num_e: int):
    r"""Return a random graph with ``num_v`` vertices and ``num_e`` edges. Edges are drawn uniformly from the set of possible edges.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``num_e`` (``int``): The Number of edges.

    Examples:
        >>> import easygraph.randomhypergraph as rh
        >>> g = rh.graph_Gnm(4, 5)
        >>> g.e
        ([(1, 2), (0, 3), (2, 3), (0, 2), (1, 3)], [1.0, 1.0, 1.0, 1.0, 1.0])
    """
    assert num_v > 1, "num_v must be greater than 1"
    assert (
        num_e < num_v * (num_v - 1) // 2
    ), "the specified num_e is larger than the possible number of edges"

    v_list = list(range(num_v))
    cur_num_e, e_set = 0, set()
    while cur_num_e < num_e:
        v = random.choice(v_list)
        w = random.choice(v_list)
        if v > w:
            v, w = w, v
        if v == w or (v, w) in e_set:
            continue
        e_set.add((v, w))
        cur_num_e += 1
    g = Graph()
    g.add_nodes(list(range(0, num_v)))
    for ee in list(e_set):
        g.add_edge(ee[0], ee[1], weight=1.0)

    return g
