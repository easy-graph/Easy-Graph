from itertools import chain

import easygraph as eg

from easygraph.utils import *


__all__ = [
    "set_edge_attributes",
    "add_path",
    "set_node_attributes",
    "selfloop_edges",
    "topological_sort",
    "number_of_selfloops",
    "density",
]


def set_edge_attributes(G, values, name=None):
    """Sets edge attributes from a given value or dictionary of values.

    .. Warning:: The call order of arguments `values` and `name`
        switched between v1.x & v2.x.

    Parameters
    ----------
    G : EasyGraph Graph

    values : scalar value, dict-like
        What the edge attribute should be set to.  If `values` is
        not a dictionary, then it is treated as a single attribute value
        that is then applied to every edge in `G`.  This means that if
        you provide a mutable object, like a list, updates to that object
        will be reflected in the edge attribute for each edge.  The attribute
        name will be `name`.

        If `values` is a dict or a dict of dict, it should be keyed
        by edge tuple to either an attribute value or a dict of attribute
        key/value pairs used to update the edge's attributes.
        For multigraphs, the edge tuples must be of the form ``(u, v, key)``,
        where `u` and `v` are nodes and `key` is the edge key.
        For non-multigraphs, the keys must be tuples of the form ``(u, v)``.

    name : string (optional, default=None)
        Name of the edge attribute to set if values is a scalar.

    Examples
    --------
    After computing some property of the edges of a graph, you may want
    to assign a edge attribute to store the value of that property for
    each edge::

        >>> G = eg.path_graph(3)
        >>> bb = eg.edge_betweenness_centrality(G, normalized=False)
        >>> eg.set_edge_attributes(G, bb, "betweenness")
        >>> G.edges[1, 2]["betweenness"]
        2.0

    If you provide a list as the second argument, updates to the list
    will be reflected in the edge attribute for each edge::

        >>> labels = []
        >>> eg.set_edge_attributes(G, labels, "labels")
        >>> labels.append("foo")
        >>> G.edges[0, 1]["labels"]
        ['foo']
        >>> G.edges[1, 2]["labels"]
        ['foo']

    If you provide a dictionary of dictionaries as the second argument,
    the entire dictionary will be used to update edge attributes::

        >>> G = eg.path_graph(3)
        >>> attrs = {(0, 1): {"attr1": 20, "attr2": "nothing"}, (1, 2): {"attr2": 3}}
        >>> eg.set_edge_attributes(G, attrs)
        >>> G[0][1]["attr1"]
        20
        >>> G[0][1]["attr2"]
        'nothing'
        >>> G[1][2]["attr2"]
        3

    Note that if the dict contains edges that are not in `G`, they are
    silently ignored::

        >>> G = eg.Graph([(0, 1)])
        >>> eg.set_edge_attributes(G, {(1, 2): {"weight": 2.0}})
        >>> (1, 2) in G.edges()
        False

    """
    if name is not None:
        # `values` does not contain attribute names
        try:
            # if `values` is a dict using `.items()` => {edge: value}
            if G.is_multigraph():
                for (u, v, key), value in values.items():
                    try:
                        G[u][v][key][name] = value
                    except KeyError:
                        pass
            else:
                for (u, v), value in values.items():
                    try:
                        G[u][v][name] = value
                    except KeyError:
                        pass
        except AttributeError:
            # treat `values` as a constant
            for u, v, data in G.edges:
                data[name] = values
    else:
        # `values` consists of doct-of-dict {edge: {attr: value}} shape
        if G.is_multigraph():
            for (u, v, key), d in values.items():
                try:
                    G[u][v][key].update(d)
                except KeyError:
                    pass
        else:
            for (u, v), d in values.items():
                try:
                    G[u][v].update(d)
                except KeyError:
                    pass


def add_path(G_to_add_to, nodes_for_path, **attr):
    """Add a path to the Graph G_to_add_to.

    Parameters
    ----------
    G_to_add_to : graph
        A EasyGraph graph
    nodes_for_path : iterable container
        A container of nodes.  A path will be constructed from
        the nodes (in order) and added to the graph.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to every edge in path.

    See Also
    --------
    add_star, add_cycle

    Examples
    --------
    >>> G = eg.Graph()
    >>> eg.add_path(G, [0, 1, 2, 3])
    >>> eg.add_path(G, [10, 11, 12], weight=7)
    """
    nlist = iter(nodes_for_path)
    try:
        first_node = next(nlist)
    except StopIteration:
        return
    G_to_add_to.add_node(first_node)
    G_to_add_to.add_edges_from(pairwise(chain((first_node,), nlist)), **attr)


def set_node_attributes(G, values, name=None):
    """Sets node attributes from a given value or dictionary of values.

    .. Warning:: The call order of arguments `values` and `name`
        switched between v1.x & v2.x.

    Parameters
    ----------
    G : EasyGraph Graph

    values : scalar value, dict-like
        What the node attribute should be set to.  If `values` is
        not a dictionary, then it is treated as a single attribute value
        that is then applied to every node in `G`.  This means that if
        you provide a mutable object, like a list, updates to that object
        will be reflected in the node attribute for every node.
        The attribute name will be `name`.

        If `values` is a dict or a dict of dict, it should be keyed
        by node to either an attribute value or a dict of attribute key/value
        pairs used to update the node's attributes.

    name : string (optional, default=None)
        Name of the node attribute to set if values is a scalar.

    Examples
    --------
    After computing some property of the nodes of a graph, you may want
    to assign a node attribute to store the value of that property for
    each node::

        >>> G = eg.path_graph(3)
        >>> bb = eg.betweenness_centrality(G)
        >>> isinstance(bb, dict)
        True
        >>> eg.set_node_attributes(G, bb, "betweenness")
        >>> G.nodes[1]["betweenness"]
        1.0

    If you provide a list as the second argument, updates to the list
    will be reflected in the node attribute for each node::

        >>> G = eg.path_graph(3)
        >>> labels = []
        >>> eg.set_node_attributes(G, labels, "labels")
        >>> labels.append("foo")
        >>> G.nodes[0]["labels"]
        ['foo']
        >>> G.nodes[1]["labels"]
        ['foo']
        >>> G.nodes[2]["labels"]
        ['foo']

    If you provide a dictionary of dictionaries as the second argument,
    the outer dictionary is assumed to be keyed by node to an inner
    dictionary of node attributes for that node::

        >>> G = eg.path_graph(3)
        >>> attrs = {0: {"attr1": 20, "attr2": "nothing"}, 1: {"attr2": 3}}
        >>> eg.set_node_attributes(G, attrs)
        >>> G.nodes[0]["attr1"]
        20
        >>> G.nodes[0]["attr2"]
        'nothing'
        >>> G.nodes[1]["attr2"]
        3
        >>> G.nodes[2]
        {}

    Note that if the dictionary contains nodes that are not in `G`, the
    values are silently ignored::

        >>> G = eg.Graph()
        >>> G.add_node(0)
        >>> eg.set_node_attributes(G, {0: "red", 1: "blue"}, name="color")
        >>> G.nodes[0]["color"]
        'red'
        >>> 1 in G.nodes
        False

    """
    # Set node attributes based on type of `values`
    if name is not None:  # `values` must not be a dict of dict
        try:  # `values` is a dict
            for n, v in values.items():
                try:
                    G.nodes[n][name] = values[n]
                except KeyError:
                    pass
        except AttributeError:  # `values` is a constant
            for n in G:
                G.nodes[n][name] = values
    else:  # `values` must be dict of dict
        for n, d in values.items():
            try:
                G.nodes[n].update(d)
            except KeyError:
                pass


def topological_generations(G):
    if not G.is_directed():
        raise AssertionError("Topological sort not defined on undirected graphs.")
    indegree_map = {v: d for v, d in G.in_degree().items() if d > 0}
    zero_indegree = [v for v, d in G.in_degree().items() if d == 0]
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
        raise AssertionError("Graph contains a cycle or graph changed during iteration")


def topological_sort(G):
    for generation in topological_generations(G):
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


@hybrid("cpp_density")
def density(G):
    r"""Returns the density of a graph.

    The density for undirected graphs is

    .. math::

       d = \frac{2m}{n(n-1)},

    and for directed graphs is

    .. math::

       d = \frac{m}{n(n-1)},

    where `n` is the number of nodes and `m`  is the number of edges in `G`.

    Notes
    -----
    The density is 0 for a graph without edges and 1 for a complete graph.
    The density of multigraphs can be higher than 1.

    Self loops are counted in the total number of edges so graphs with self
    loops can have density higher than 1.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == 0 or n <= 1:
        return 0
    d = m / (n * (n - 1))
    if not G.is_directed():
        d *= 2
    return d
