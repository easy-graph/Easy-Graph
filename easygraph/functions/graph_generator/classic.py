import itertools

from easygraph.classes.graph import Graph
from easygraph.utils import nodes_or_number
from easygraph.utils import pairwise


__all__ = ["empty_graph", "path_graph", "complete_graph"]


@nodes_or_number(0)
def empty_graph(n=0, create_using=None, default=Graph):
    if create_using is None:
        G = default()
    elif hasattr(create_using, "_adj"):
        # create_using is a EasyGraph style Graph
        G = create_using
    else:
        # try create_using as constructor
        G = create_using()

    n_name, nodes = n
    G.add_nodes_from(nodes)
    return G


@nodes_or_number(0)
def path_graph(n, create_using=None):
    n_name, nodes = n
    G = empty_graph(nodes, create_using)
    G.add_edges_from(pairwise(nodes))
    return G


@nodes_or_number(0)
def complete_graph(n, create_using=None):
    """Return the complete graph `K_n` with n nodes.

    A complete graph on `n` nodes means that all pairs
    of distinct nodes have an edge connecting them.

    Parameters
    ----------
    n : int or iterable container of nodes
        If n is an integer, nodes are from range(n).
        If n is a container of nodes, those nodes appear in the graph.
    create_using : EasyGraph graph constructor, optional (default=eg.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Examples
    --------
    >>> G = eg.complete_graph(9)
    >>> len(G)
    9
    >>> G.size()
    36
    >>> G = eg.complete_graph(range(11, 14))
    >>> list(G.nodes())
    [11, 12, 13]
    >>> G = eg.complete_graph(4, eg.DiGraph())
    >>> G.is_directed()
    True

    """
    n_name, nodes = n
    G = empty_graph(n_name, create_using)
    if len(nodes) > 1:
        if G.is_directed():
            edges = itertools.permutations(nodes, 2)
        else:
            edges = itertools.combinations(nodes, 2)
        G.add_edges_from(edges)
    return G
