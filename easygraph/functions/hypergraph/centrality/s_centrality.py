import easygraph as eg


__all__ = ["s_betweenness", "s_closeness", "s_eccentricity"]


def s_betweenness(H, s=1, weight=False, n_workers=None):
    """Computes the betweenness centrality for each edge in the hypergraph.

    Computes the betweenness centrality for each edge in the hypergraph.

    Parameters
    ----------
    H : eg.Hypergraph.
        The hypergraph to compute

    s : int, optional.

    Returns
    ----------
    dict
    The keys are the edges and the values are the betweenness centrality.
    The betweenness centrality for each edge in the hypergraph.


    """

    linegraph = H.get_linegraph(s=s, weight=weight)
    results = eg.betweenness_centrality(linegraph, n_workers=n_workers)
    return results


def s_closeness(H, s=1, weight=False, n_workers=None):
    """
    Compute the closeness centrality for each edge in the hypergraph.

    Parameters
    ----------
    H : eg.Hypergraph.
    s : int, optional

    Returns
    -------
    dict. The closeness centrality for each edge in the hypergraph. The keys are the edges and the values are the closeness centrality.
    """
    linegraph = H.get_linegraph(s=s, weight=weight)
    results = eg.closeness_centrality(linegraph, n_workers=n_workers)
    return results


def s_eccentricity(H, s=1, edges=True, source=None):
    r"""
    The length of the longest shortest path from a vertex $u$ to every other vertex in
    the s-linegraph.
    $V$ = set of vertices in the s-linegraph
    $d$ = shortest path distance

    .. math::

        \text{s-ecc}(u) = \text{max}\{d(u,v): v \in V\}

    Parameters
    ----------
    H : eg.Hypergraph

    s : int, optional

    edges : bool, optional
        Indicates if method should compute edge linegraph (default) or node linegraph.

    source : str, optional
        Identifier of node or edge of interest for computing centrality

    Returns
    -------
    dict or float
        returns the s-eccentricity value of the edges(nodes).
        If source=None a dictionary of values for each s-edge in H is returned.
        If source then a single value is returned.
        If the s-linegraph is disconnected, np.inf is returned.

    """

    g = H.get_linegraph(s=s)
    result = eg.eccentricity(g)
    if source:
        return result[source]
    else:
        return result
