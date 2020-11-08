import easygraph as eg
import math


def sum_of_shortest_paths(G, S):
    """Returns the difference between the sum of lengths of all pairs shortest paths in G and the one in G\S.
    The experiment metrics in [1]_

    Parameters
    ----------
    G: easygraph.Graph or easygraph.DiGraph

    S: list of int
        A list of nodes witch are structural hole spanners.

    Returns
    -------
    differ_between_sum : int
        The difference between the sum of lengths of all pairs shortest paths in G and the one in G\S.
        C(G/S)-C(G)

    Examples
    --------
    >>> G_t=eg.datasets.get_graph_blogcatalog()
    >>> S_t=eg.AP_Greedy(G_t, 10000)
    >>> diff = sum_of_shortest_paths(G_t, S_t)
    >>> print(diff)

    References
    ----------
    .. [1] https://dl.acm.org/profile/81484650642

    """
    mat_G = eg.Floyd(G)
    sum_G = 0
    inf_const_G=math.ceil((G.number_of_nodes()**3)/3)
    for i in mat_G.values():
        for j in i.values():
            if math.isinf(j):
                j=inf_const_G
            sum_G += j
    G_S = G.copy()
    G_S.remove_nodes(S)
    mat_G_S = eg.Floyd(G_S)
    sum_G_S = 0
    inf_const_G_S = math.ceil((G_S.number_of_nodes() ** 3) / 3)
    for i in mat_G_S.values():
        for j in i.values():
            if math.isinf(j):
                j = inf_const_G_S
            sum_G_S += j
    return sum_G_S - sum_G


def nodes_of_max_cc_without_shs(G, S):
    """Returns the number of nodes in the maximum connected component in graph G\S.
    The experiment metrics in [1]_

    Parameters
    ----------
    G: easygraph.Graph or easygraph.DiGraph

    S: list of int
        A list of nodes witch are structural hole spanners.

    Returns
    -------
    G_S_nodes_of_max_CC: int
        The number of nodes in the maximum connected component in graph G\S.

    Examples
    --------
    >>> G_t=eg.datasets.get_graph_blogcatalog()
    >>> S_t=eg.AP_Greedy(G_t, 10000)
    >>> maxx = nodes_of_max_cc_without_shs(G_t, S_t)
    >>> print(maxx)

    References
    ----------
    .. [1] https://dl.acm.org/profile/81484650642

    """
    G_S = G.copy()
    G_S.remove_nodes(S)
    ccs = eg.connected_components(G_S)
    max_num = 0
    for cc in ccs:
        if len(cc) > max_num:
            max_num = len(cc)
    return max_num
