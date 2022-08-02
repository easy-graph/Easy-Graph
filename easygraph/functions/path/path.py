from easygraph.utils.decorators import *


__all__ = [
    "Dijkstra",
    "Floyd",
    "Prim",
    "Kruskal",
    "single_source_bfs",
    "single_source_dijkstra",
    "multi_source_dijkstra",
]

try:
    from cpp_easygraph import cpp_dijkstra_multisource
    from cpp_easygraph import cpp_Floyd
    from cpp_easygraph import cpp_Kruskal
    from cpp_easygraph import cpp_Prim
except ImportError:
    pass


@not_implemented_for("multigraph")
def Dijkstra(G, node):
    """Returns the length of paths from the certain node to remaining nodes

    Parameters
    ----------
    G : graph
        weighted graph
    node : int

    Returns
    -------
    result_dict : dict
        the length of paths from the certain node to remaining nodes

    Examples
    --------
    Returns the length of paths from node 1 to remaining nodes

    >>> Dijkstra(G,node=1)

    """
    return single_source_dijkstra(G, node)


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
def Floyd(G):
    """Returns the length of paths from all nodes to remaining nodes

    Parameters
    ----------
    G : graph
        weighted graph

    Returns
    -------
    result_dict : dict
        the length of paths from all nodes to remaining nodes

    Examples
    --------
    Returns the length of paths from all nodes to remaining nodes

    >>> Floyd(G)

    """
    if G.cflag == 1:
        return cpp_Floyd(G)
    adj = G.adj.copy()
    result_dict = {}
    for i in G:
        result_dict[i] = {}
    for i in G:
        temp_key = adj[i].keys()
        for j in G:
            if j in temp_key:
                result_dict[i][j] = adj[i][j].get("weight", 1)
            else:
                result_dict[i][j] = float("inf")
            if i == j:
                result_dict[i][i] = 0
    for k in G:
        for i in G:
            for j in G:
                temp = result_dict[i][k] + result_dict[k][j]
                if result_dict[i][j] > temp:
                    result_dict[i][j] = temp
    return result_dict


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
def Prim(G):
    """Returns the edges that make up the minimum spanning tree

    Parameters
    ----------
    G : graph
        weighted graph

    Returns
    -------
    result_dict : dict
        the edges that make up the minimum spanning tree

    Examples
    --------
    Returns the edges that make up the minimum spanning tree

    >>> Prim(G)

    """
    if G.cflag == 1:
        return cpp_Prim(G)
    adj = G.adj.copy()
    result_dict = {}
    for i in G:
        result_dict[i] = {}
    selected = []
    candidate = []
    for i in G:
        if not selected:
            selected.append(i)
        else:
            candidate.append(i)
    while len(candidate):
        start = None
        end = None
        min_weight = float("inf")
        for i in selected:
            for j in candidate:
                if i in G and j in G[i] and adj[i][j].get("weight", 1) < min_weight:
                    start = i
                    end = j
                    min_weight = adj[i][j].get("weight", 1)
        if start != None and end != None:
            result_dict[start][end] = min_weight
            selected.append(end)
            candidate.remove(end)
        else:
            break
    return result_dict


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
def Kruskal(G):
    """Returns the edges that make up the minimum spanning tree

    Parameters
    ----------
    G : graph
        weighted graph

    Returns
    -------
    result_dict : dict
        the edges that make up the minimum spanning tree

    Examples
    --------
    Returns the edges that make up the minimum spanning tree

    >>> Kruskal(G)

    """
    if G.cflag == 1:
        return cpp_Kruskal(G)
    adj = G.adj.copy()
    result_dict = {}
    edge_list = []
    for i in G:
        result_dict[i] = {}
    for i in G:
        for j in G[i]:
            weight = adj[i][j].get("weight", 1)
            edge_list.append([i, j, weight])
    edge_list.sort(key=lambda a: a[2])
    group = [[i] for i in G]
    for edge in edge_list:
        for i in range(len(group)):
            if edge[0] in group[i]:
                m = i
            if edge[1] in group[i]:
                n = i
        if m != n:
            result_dict[edge[0]][edge[1]] = edge[2]
            group[m] = group[m] + group[n]
            group[n] = []
    return result_dict


@not_implemented_for("multigraph")
def single_source_bfs(G, source, target=None):
    nextlevel = {source: 0}
    return dict(_single_source_bfs(G.adj, nextlevel, target=target))


def _single_source_bfs(adj, firstlevel, target=None):
    seen = {}
    level = 0
    nextlevel = firstlevel

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            if v not in seen:
                seen[v] = level
                nextlevel.update(adj[v])
                yield (v, level)
                if v == target:
                    break
        level += 1
    del seen


@not_implemented_for("multigraph")
def single_source_dijkstra(G, source, weight="weight", target=None):
    return multi_source_dijkstra(G, {source}, weight, target=target)


@not_implemented_for("multigraph")
def multi_source_dijkstra(G, sources, weight="weight", target=None):
    return _dijkstra_multisource(G, sources, weight, target=target)


def _dijkstra_multisource(G, sources, weight="weight", target=None):
    if G.cflag == 1:
        return cpp_dijkstra_multisource(G, sources, weight, target)
    from heapq import heappop
    from heapq import heappush

    push = heappush
    pop = heappop
    adj = G.adj
    dist = {}
    seen = {}
    from itertools import count

    c = count()
    Q = []
    for source in sources:
        seen[source] = 0
        push(Q, (0, next(c), source))
    while Q:
        (d, _, v) = pop(Q)
        if v in dist:
            continue
        dist[v] = d
        if v == target:
            break
        for u in adj[v]:
            cost = adj[v][u].get(weight, 1)
            vu_dist = dist[v] + cost
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError("Contradictory paths found:", "negative weights?")
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(Q, (vu_dist, next(c), u))
            else:
                continue
    return dist
