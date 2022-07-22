from itertools import product

from easygraph.utils import *


__all__ = ["modularity"]


@not_implemented_for("multigraph")
def modularity(G, communities, weight="weight"):
    r"""
    Returns the modularity of the given partition of the graph.
    Modularity is defined in [1]_ as

    .. math::

        Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_ik_j}{2m}\right)
            \delta(c_i,c_j)

    where m is the number of edges, A is the adjacency matrix of
    `G`,

    .. math::

        k_i\ is\ the\ degree\ of\ i\ and\ \delta(c_i, c_j)\ is\ 1\ if\ i\ and\ j\ are\ in\ the\ same\ community\ and\ 0\ otherwise.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    communities : list or iterable of set of nodes
        These node sets must represent a partition of G's nodes.

    weight : string, optional (default : 'weight')
        The key for edge weight.

    Returns
    ----------
    Q : float
        The modularity of the partition.

    References
    ----------
    .. [1] M. E. J. Newman *Networks: An Introduction*, page 224.
       Oxford University Press, 2011.

    """
    # TODO: multigraph not included.

    if not isinstance(communities, list):
        communities = list(communities)

    directed = G.is_directed()
    m = G.size(weight=weight)
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        norm = 1 / m
    else:
        out_degree = dict(G.degree(weight=weight))
        in_degree = out_degree
        norm = 1 / (2 * m)

    def val(u, v):
        try:
            w = G[u][v].get(weight, 1)
        except KeyError:
            w = 0
        # Double count self-loops if the graph is undirected.
        if u == v and not directed:
            w *= 2
        return w - in_degree[u] * out_degree[v] * norm

    Q = sum(val(u, v) for c in communities for u, v in product(c, repeat=2))
    return Q * norm
