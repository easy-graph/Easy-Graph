import easygraph as eg
import easygraph.functions.path


__all__ = ["diameter", "eccentricity"]


def eccentricity(G, v=None, sp=None):
    """Returns the eccentricity of nodes in G.

    The eccentricity of a node v is the maximum distance from v to
    all other nodes in G.

    Parameters
    ----------
    G : EasyGraph graph
       A graph

    v : node, optional
       Return value of specified node

    sp : dict of dicts, optional
       All pairs shortest path lengths as a dictionary of dictionaries

    Returns
    -------
    ecc : dictionary
       A dictionary of eccentricity values keyed by node.

    Examples
    --------
    >>> G = eg.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> dict(eg.eccentricity(G))
    {1: 2, 2: 3, 3: 2, 4: 2, 5: 3}

    >>> dict(eg.eccentricity(G, v=[1, 5]))  # This returns the eccentrity of node 1 & 5
    {1: 2, 5: 3}

    """
    #    if v is None:                # none, use entire graph
    #        nodes=G.nodes()
    #    elif v in G:               # is v a single node
    #        nodes=[v]
    #    else:                      # assume v is a container of nodes
    #        nodes=v
    order = G.order()

    e = {}
    for n in G.nbunch_iter(v):
        if sp is None:
            length = eg.single_source_dijkstra(G, n)
            L = len(length)
        else:
            try:
                length = sp[n]
                L = len(length)
            except TypeError as err:
                raise eg.EasyGraphError('Format of "sp" is invalid.') from err
        if L != order:
            if G.is_directed():
                msg = (
                    "Found infinite path length because the digraph is not"
                    " strongly connected"
                )
            else:
                msg = "Found infinite path length because the graph is not connected"
            raise eg.EasyGraphError(msg)

        e[n] = max(length.values())

    if v in G:
        return e[v]  # return single value
    else:
        return e


def diameter(G, e=None):
    """Returns the diameter of the graph G.

    The diameter is the maximum eccentricity.

    Parameters
    ----------
    G : EasyGraph graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    Returns
    -------
    d : integer
       Diameter of graph

    Examples
    --------
    >>> G = eg.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> eg.diameter(G)
    3

    See Also
    --------
    eccentricity
    """

    if e is None:
        e = eccentricity(G)
    return max(e.values())
