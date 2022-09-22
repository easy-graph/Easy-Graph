__all__ = [
    "average_degree",
]


def average_degree(G) -> float:
    """Returns the average degree of the graph.

    Parameters
    ----------
    G : graph
        A EasyGraph graph

    Returns
    -------
    average degree : float
        The average degree of the graph.

    Notes
    -----
    Self loops are counted twice in the total degree of a node.

    Examples
    --------
    >>> G = eg.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    >>> G.add_edge(1, 2)
    >>> G.add_edge(2, 3)
    >>> eg.average_degree(G)
    1.3333333333333333
    """
    return G.number_of_edges() / G.number_of_nodes() * 2
