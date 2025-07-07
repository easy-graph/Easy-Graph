import warnings

import easygraph as eg

from easygraph.functions.path.path import *


def average_shortest_path_length(G, weight=None, method=None):
    r"""Returns the average shortest path length.

    The average shortest path length is

    .. math::

       a =\sum_{\substack{s,t \in V \\ s\neq t}} \frac{d(s, t)}{n(n-1)}

    where `V` is the set of nodes in `G`,
    `d(s, t)` is the shortest path from `s` to `t`,
    and `n` is the number of nodes in `G`.

    .. versionchanged:: 3.0
       An exception is raised for directed graphs that are not strongly
       connected.

    Parameters
    ----------
    G : EasyGraph graph

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'unweighted' or 'dijkstra')
        The algorithm to use to compute the path lengths.
        Supported options are 'unweighted', 'dijkstra', 'bellman-ford',
        'floyd-warshall' and 'floyd-warshall-numpy'.
        Other method values produce a ValueError.
        The default method is 'unweighted' if `weight` is None,
        otherwise the default method is 'dijkstra'.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph (that is, the graph on zero nodes).

    NetworkXError
        If `G` is not connected (or not strongly connected, in the case
        of a directed graph).

    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = eg.path_graph(5)
    >>> eg.average_shortest_path_length(G)
    2.0

    For disconnected graphs, you can compute the average shortest path
    length for each component

    >>> G = eg.Graph([(1, 2), (3, 4)])
    >>> for C in (G.subgraph(c).copy() for c in eg.connected_components(G)):
    ...     print(eg.average_shortest_path_length(C))
    1.0
    1.0

    """
    single_source_methods = ["single_source_bfs", "dijkstra"]
    all_pairs_methods = ["Floyed"]
    supported_methods = single_source_methods + all_pairs_methods

    if method is None:
        method = "single_source_bfs" if weight is None else "dijkstra"
    if method not in supported_methods:
        raise ValueError(f"method not supported: {method}")

    n = len(G)
    # For the special case of the null graph, raise an exception, since
    # there are no paths in the null graph.
    if n == 0:
        msg = (
            "the null graph has no paths, thus there is no average shortest path length"
        )
        raise eg.EasyGraphPointlessConcept(msg)
    # For the special case of the trivial graph, return zero immediately.
    if n == 1:
        return 0
    # Shortest path length is undefined if the graph is not strongly connected.
    if G.is_directed() and not eg.is_strongly_connected(G):
        raise eg.EasyGraphError("Graph is not strongly connected.")
    # Shortest path length is undefined if the graph is not connected.
    if not G.is_directed() and not eg.is_connected(G):
        raise eg.EasyGraphError("Graph is not connected.")

    # Compute all-pairs shortest paths.
    def path_length(v):
        if method == "single_source_bfs":
            return eg.single_source_bfs(G, v)
        elif method == "dijkstra":
            return eg.Dijkstra(G, v, weight=weight)

    if method in single_source_methods:
        # Sum the distances for each (ordered) pair of source and target node.
        s = sum(l for u in G for l in path_length(u).values())
    else:
        all_pairs = eg.Floyed(G, weight=weight)
        s = sum(sum(t.values()) for t in all_pairs.values())
    return s / (n * (n - 1))
