import easygraph as eg

from easygraph.utils import *


__all__ = [
    "weakTie",
    "weakTieLocal",
]


def _computeTieStrength(G, node_u, node_v):
    F_u = set(G.neighbors(node=node_u))
    F_u.add(node_u)
    F_v = set(G.neighbors(node=node_v))
    F_v.add(node_v)
    uni = len(F_u.union(F_v))
    inter = len(F_u.intersection(F_v))
    S_uv = inter / uni
    G[node_u][node_v]["strength"] = S_uv


def _computeAllTieStrength(G):
    for edge in G.edges:
        node_u = edge[0]
        node_v = edge[1]
        _computeTieStrength(G, node_u, node_v)
    # print(G.edges)


def _strongly_connected_components(G, threshold):
    """Generate nodes in strongly connected components of graph with constraint threshold.

    Parameters
    ----------
    G : easygraph.DiGraph
        A directed graph.

    threshold: float
        the edge whose tie strength is smaller than threshold will be ignored.

    Returns
    -------
    comp : generator of sets
        A generator of sets of nodes, one for each strongly connected
        component of G.

    Examples
    --------
    # >>> _strongly_connected_components(G, 0.2)

    Notes
    -----
    Uses Tarjan's algorithm[1]_ with Nuutila's modifications[2]_.
    Nonrecursive version of algorithm.

    References
    ----------
    .. [1] Depth-first search and linear graph algorithms, R. Tarjan
       SIAM Journal of Computing 1(2):146-160, (1972).

    .. [2] On finding the strongly connected components in a directed graph.
       E. Nuutila and E. Soisalon-Soinen
       Information Processing Letters 49(1): 9-14, (1994)..

    """
    preorder = {}
    lowlink = {}
    scc_found = set()
    scc_queue = []
    i = 0  # Preorder counter
    for source in G:
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                done = True
                for w in G[v]:
                    if G[v][w]["strength"] >= threshold:
                        if w not in preorder:
                            queue.append(w)
                            done = False
                            break
                if done:
                    lowlink[v] = preorder[v]
                    for w in G[v]:
                        if G[v][w]["strength"] >= threshold:
                            if w not in scc_found:
                                if preorder[w] > preorder[v]:
                                    lowlink[v] = min([lowlink[v], lowlink[w]])
                                else:
                                    lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
                            scc.add(k)
                        scc_found.update(scc)
                        yield scc
                    else:
                        scc_queue.append(v)


def _computeCloseness(G, c, u, threshold, length):
    n = 0
    strength_sum_u = 0
    for v in c:
        if u in G[v] and v != u:
            if G[v][u]["strength"] != 0:
                n += 1
                strength_sum_u += G[v][u]["strength"]
    closeness_c_u = (strength_sum_u - n * threshold) / length
    return closeness_c_u


def _computeScore(G, threshold):
    score_dict = {}
    for node in G.nodes:
        score_dict[node] = 0
    for c in _strongly_connected_components(G, threshold):
        length = len(c)
        for u in G.nodes:
            closeness_c_u = _computeCloseness(G, c, u, threshold, length)
            if closeness_c_u < 0:
                score_dict[u] += (-1) * closeness_c_u
    return score_dict


@not_implemented_for("multigraph")
def weakTie(G, threshold, k):
    """Return top-k nodes with highest scores which were computed by WeakTie method.

    Parameters
    ----------
    G: easygraph.DiGraph

    k: int
        top - k nodes with highest scores.

    threshold: float
        tie strength threshold.

    Returns
    -------
    SHS_list : list
        The list of each nodes with highest scores.

    score_dict: dict
        The score of each node, can be used for WeakTie-Local and WeakTie-Bi.

    See Also
    -------
    weakTieLocal

    Examples
    --------
    # >>> SHS_list,score_dict=weakTie(G, 0.2, 3)

    References
    ----------
    .. [1] Mining Brokers in Dynamic Social Networks. Chonggang Song, Wynne Hsu, Mong Li Lee. Proc. of ACM CIKM, 2015.

    """
    _computeAllTieStrength(G)
    score_dict = _computeScore(G, threshold)
    ordered_set = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    SHS_list = []
    for i in range(k):
        SHS_list.append((ordered_set[i])[0])
    print("score dict:", score_dict)
    print("top-k nodes:", SHS_list)
    return SHS_list, score_dict


@not_implemented_for("multigraph")
def _updateScore(u, G, threshold):
    score_u = 0
    for c in _strongly_connected_components(G, threshold):
        length = len(c)
        closeness_c_u = _computeCloseness(G, c, u, threshold, length)
        if closeness_c_u < 0:
            score_u -= closeness_c_u
    return score_u


def _get2hop(G, node):
    neighbors = []
    firstlevel = {node: 1}
    seen = {}  # level (number of hops) when seen in BFS
    level = 0  # the current level
    nextlevel = set(firstlevel)  # set of nodes to check at next level
    n = len(G.adj)
    while nextlevel and level <= 2:
        thislevel = nextlevel  # advance to next level
        nextlevel = set()  # and start a new set (fringe)
        found = []
        for v in thislevel:
            if v not in seen:
                seen[v] = level  # set the level of vertex v
                found.append(v)
                # yield (v, level)
                neighbors.append(v)
        if len(seen) == n:
            return
        for v in found:
            nextlevel.update(G.adj[v])
        level += 1
    del seen
    return neighbors


def _commonUpdate(G, node_u, node_v, threshold, score_dict):
    for node_w in G.neighbors(node=node_u):
        _computeTieStrength(G, node_u, node_w)
    for node_w in G.predecessors(node=node_u):
        _computeTieStrength(G, node_w, node_u)
    G_un = eg.Graph()
    for node in G.nodes:
        G_un.add_node(node)
    for edge in G.edges:
        if not G_un.has_edge(edge[0], edge[1]):
            G_un.add_edge(edge[0], edge[1])
    u_2hop = _get2hop(G_un, node_u)
    G_u = G.nodes_subgraph(from_nodes=u_2hop)
    v_2hop = _get2hop(G_un, node_v)
    G_v = G.nodes_subgraph(from_nodes=v_2hop)
    score_u = _updateScore(node_u, G_u, threshold)
    score_v = _updateScore(node_v, G_v, threshold)
    score_dict[node_u] = score_u
    score_dict[node_v] = score_v
    all_neigh_u = list(set(G.all_neighbors(node=node_u)))
    # print("all_neigh:",all_neigh_u)
    all_neigh_v = list(set(G.all_neighbors(node=node_v)))
    for node_w in all_neigh_u:
        if node_w in all_neigh_v:
            w_2hop = _get2hop(G_un, node_w)
            G_w = G.nodes_subgraph(from_nodes=w_2hop)
            score_w = _updateScore(node_w, G_w, threshold)
        else:
            score_w = 0
            w_2hop = _get2hop(G_un, node_w)
            G_w = G.nodes_subgraph(from_nodes=w_2hop)
            for c in _strongly_connected_components(G_w, threshold):
                if node_u in c:
                    length = len(c)
                    closeness_c_w = _computeCloseness(G, c, node_w, threshold, length)
                    if closeness_c_w < 0:
                        score_w -= closeness_c_w
        score_dict[node_w] = score_w


def weakTieLocal(G, edges_plus, edges_delete, threshold, score_dict, k):
    """Find brokers in evolving social networks, utilize the 2-hop neighborhood of an affected node to identify brokers.

    Parameters
    ----------
    G: easygraph.DiGraph

    edges_plus: list of list
        set of edges to be added

    edges_delete: list of list
        set of edges to be removed

    threshold: float
        tie strength threshold.

    score_dict: dict
        The score of each node computed before.

    k: int
        top - k nodes with highest scores.

    Returns
    -------
    SHS_list : list
        The list of each nodes with highest scores.

    See Also
    -------
    weakTie

    Examples
    --------
    # >>> SHS_list=weakTieLocal(G, [[2, 7]], [[1,3]], 0.2, score_dict, 3)

    References
    ----------
    .. [1] Mining Brokers in Dynamic Social Networks. Chonggang Song, Wynne Hsu, Mong Li Lee. Proc. of ACM CIKM, 2015.

    """
    for edge in edges_plus:
        G.add_edge(edge[0], edge[1])
        _computeTieStrength(G, edge[0], edge[1])
        _commonUpdate(G, edge[0], edge[1], threshold, score_dict)
    for edge in edges_delete:
        G.remove_edge(edge[0], edge[1])
        _commonUpdate(G, edge[0], edge[1], threshold, score_dict)
    ordered_set = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    SHS_list = []
    for i in range(k):
        SHS_list.append((ordered_set[i])[0])
    print("updated score:", score_dict)
    print("top-k nodes:", SHS_list)
    return SHS_list


if __name__ == "__main__":
    G = eg.DiGraph()
    G.add_edge(1, 5)
    G.add_edge(1, 4)
    G.add_edge(2, 1)
    G.add_edge(2, 6)
    G.add_edge(2, 9)
    G.add_edge(3, 4)
    G.add_edge(3, 1)
    G.add_edge(4, 3)
    G.add_edge(4, 1)
    G.add_edge(4, 5)
    G.add_edge(5, 4)
    G.add_edge(5, 8)
    G.add_edge(6, 1)
    G.add_edge(6, 2)
    G.add_edge(7, 2)
    G.add_edge(7, 3)
    G.add_edge(7, 10)
    G.add_edge(8, 4)
    G.add_edge(8, 5)
    G.add_edge(9, 6)
    G.add_edge(9, 10)
    G.add_edge(10, 7)
    G.add_edge(10, 9)
    SHS_list, score_dict = weakTie(G, 0.2, 3)
    SHS_list = weakTieLocal(G, [[2, 7]], [[2, 7]], 0.2, score_dict, 3)
