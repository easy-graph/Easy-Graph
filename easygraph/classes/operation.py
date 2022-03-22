import easygraph as eg

__all__ = [
    "selfloop_edges", 
    "topological_sort",
    "number_of_selfloops",
]

def topological_generations(G):
    if not G.is_directed():
        raise AssertionError("Topological sort not defined on undirected graphs.")
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [v for v, d in G.in_degree() if d == 0]
    while zero_indegree:
        this_generation = zero_indegree
        zero_indegree = []
        for node in this_generation:
            if node not in G:
                raise RuntimeError("Graph changed during iteration")
            for child in G.neighbors(node):
                try:
                    indegree_map[child] -= 1
                except KeyError as err:
                    raise RuntimeError("Graph changed during iteration") from err
                if indegree_map[child] == 0:
                    zero_indegree.append(child)
                    del indegree_map[child]
        yield this_generation

    if indegree_map:
        raise AssertionError(
            "Graph contains a cycle or graph changed during iteration"
        )

def topological_sort(G):
    for generation in eg.topological_generations(G):
        yield from generation


def number_of_selfloops(G):
    """Returns the number of selfloop edges.

    A selfloop edge has the same node at both ends.

    Returns
    -------
    nloops : int
        The number of selfloops.

    See Also
    --------
    nodes_with_selfloops, selfloop_edges

    Examples
    --------
    >>> G = eg.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    >>> G.add_edge(1, 1)
    >>> G.add_edge(1, 2)
    >>> eg.number_of_selfloops(G)
    1
    """
    return sum(1 for _ in eg.selfloop_edges(G))

def selfloop_edges(G, data=False, keys=False, default=None):
    """Returns an iterator over selfloop edges.

    A selfloop edge has the same node at both ends.

    Parameters
    ----------
    G : graph
        A EasyGraph graph.
    data : string or bool, optional (default=False)
        Return selfloop edges as two tuples (u, v) (data=False)
        or three-tuples (u, v, datadict) (data=True)
        or three-tuples (u, v, datavalue) (data='attrname')
    keys : bool, optional (default=False)
        If True, return edge keys with each edge.
    default : value, optional (default=None)
        Value used for edges that don't have the requested attribute.
        Only relevant if data is not True or False.

    Returns
    -------
    edgeiter : iterator over edge tuples
        An iterator over all selfloop edges.

    See Also
    --------
    nodes_with_selfloops, number_of_selfloops

    Examples
    --------
    >>> G = eg.MultiGraph()  # or Graph, DiGraph, MultiDiGraph, etc
    >>> ekey = G.add_edge(1, 1)
    >>> ekey = G.add_edge(1, 2)
    >>> list(eg.selfloop_edges(G))
    [(1, 1)]
    >>> list(eg.selfloop_edges(G, data=True))
    [(1, 1, {})]
    >>> list(eg.selfloop_edges(G, keys=True))
    [(1, 1, 0)]
    >>> list(eg.selfloop_edges(G, keys=True, data=True))
    [(1, 1, 0, {})]
    """
    if data is True:
        if G.is_multigraph():
            if keys is True:
                return (
                    (n, n, k, d)
                    for n, nbrs in G.adj.items()
                    if n in nbrs
                    for k, d in nbrs[n].items()
                )
            else:
                return (
                    (n, n, d)
                    for n, nbrs in G.adj.items()
                    if n in nbrs
                    for d in nbrs[n].values()
                )
        else:
            return ((n, n, nbrs[n]) for n, nbrs in G.adj.items() if n in nbrs)
    elif data is not False:
        if G.is_multigraph():
            if keys is True:
                return (
                    (n, n, k, d.get(data, default))
                    for n, nbrs in G.adj.items()
                    if n in nbrs
                    for k, d in nbrs[n].items()
                )
            else:
                return (
                    (n, n, d.get(data, default))
                    for n, nbrs in G.adj.items()
                    if n in nbrs
                    for d in nbrs[n].values()
                )
        else:
            return (
                (n, n, nbrs[n].get(data, default))
                for n, nbrs in G.adj.items()
                if n in nbrs
            )
    else:
        if G.is_multigraph():
            if keys is True:
                return (
                    (n, n, k) for n, nbrs in G.adj.items() if n in nbrs for k in nbrs[n]
                )
            else:
                return (
                    (n, n)
                    for n, nbrs in G.adj.items()
                    if n in nbrs
                    for i in range(len(nbrs[n]))  # for easy edge removal (#4068)
                )
        else:
            return ((n, n) for n, nbrs in G.adj.items() if n in nbrs)

