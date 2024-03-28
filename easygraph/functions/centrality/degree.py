from easygraph.utils.decorators import *


__all__ = ["degree_centrality", "in_degree_centrality", "out_degree_centrality"]


@not_implemented_for("multigraph")
def degree_centrality(G):
    """Compute the degree centrality for nodes in a bipartite network.

    The degree centrality for a node v is the fraction of nodes it
    is connected to.

    parameters
    ----------
    G : graph
      A easygraph graph

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with degree centrality as the value.

    Notes
    -----
    The degree centrality are normalized by dividing by n-1 where
    n is number of nodes in G.
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in (G.degree()).items()}
    return centrality


@not_implemented_for("multigraph")
@only_implemented_for_Directed_graph
def in_degree_centrality(G):
    """Compute the in-degree centrality for nodes.

    The in-degree centrality for a node v is the fraction of nodes its
    incoming edges are connected to.

    Parameters
    ----------
    G : graph
        A EasyGraph graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with in-degree centrality as values.

    Raises
    ------
    EasyGraphNotImplemented:
        If G is undirected.

    See Also
    --------
    degree_centrality, out_degree_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.in_degree().items()}
    return centrality


@not_implemented_for("multigraph")
@only_implemented_for_Directed_graph
def out_degree_centrality(G):
    """Compute the out-degree centrality for nodes.

    The out-degree centrality for a node v is the fraction of nodes its
    outgoing edges are connected to.

    Parameters
    ----------
    G : graph
        A EasyGraph graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with out-degree centrality as values.

    Raises
    ------
    EasyGraphNotImplemented:
        If G is undirected.

    See Also
    --------
    degree_centrality, in_degree_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.out_degree().items()}
    return centrality
