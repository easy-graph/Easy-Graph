import easygraph as eg


__all__ = [
    "from_pyGraphviz_agraph",
    "to_pyGraphviz_agraph",
]


def from_pyGraphviz_agraph(A, create_using=None):
    """Returns a EasyGraph Graph or DiGraph from a PyGraphviz graph.

    Parameters
    ----------
    A : PyGraphviz AGraph
      A graph created with PyGraphviz

    create_using : EasyGraph graph constructor, optional (default=None)
       Graph type to create. If graph instance, then cleared before populated.
       If `None`, then the appropriate Graph type is inferred from `A`.

    Examples
    --------
    >>> K5 = eg.complete_graph(5)
    >>> A = eg.to_pyGraphviz_agraph(K5)
    >>> G = eg.from_pyGraphviz_agraph(A)

    Notes
    -----
    The Graph G will have a dictionary G.graph_attr containing
    the default graphviz attributes for graphs, nodes and edges.

    Default node attributes will be in the dictionary G.node_attr
    which is keyed by node.

    Edge attributes will be returned as edge data in G.  With
    edge_attr=False the edge data will be the Graphviz edge weight
    attribute or the value 1 if no edge weight attribute is found.

    """
    if create_using is None:
        if A.is_directed():
            if A.is_strict():
                create_using = eg.DiGraph
            else:
                create_using = eg.MultiDiGraph
        else:
            if A.is_strict():
                create_using = eg.Graph
            else:
                create_using = eg.MultiGraph

    # assign defaults
    N = eg.empty_graph(0, create_using)
    if A.name is not None:
        N.name = A.name

    # add graph attributes
    N.graph.update(A.graph_attr)

    # add nodes, attributes to N.node_attr
    for n in A.nodes():
        str_attr = {str(k): v for k, v in n.attr.items()}
        N.add_node(str(n), **str_attr)

    # add edges, assign edge data as dictionary of attributes
    for e in A.edges():
        u, v = str(e[0]), str(e[1])
        attr = dict(e.attr)
        str_attr = {str(k): v for k, v in attr.items()}
        if not N.is_multigraph():
            if e.name is not None:
                str_attr["key"] = e.name
            N.add_edge(u, v, **str_attr)
        else:
            N.add_edge(u, v, key=e.name, **str_attr)

    # add default attributes for graph, nodes, and edges
    # hang them on N.graph_attr
    N.graph["graph"] = dict(A.graph_attr)
    N.graph["node"] = dict(A.node_attr)
    N.graph["edge"] = dict(A.edge_attr)
    return N


def to_pyGraphviz_agraph(N):
    """Returns a pygraphviz graph from a EasyGraph graph N.

    Parameters
    ----------
    N : EasyGraph graph
      A graph created with EasyGraph

    Examples
    --------
    >>> K5 = eg.complete_graph(5)
    >>> A = eg.to_pyGraphviz_agraph(K5)

    Notes
    -----
    If N has an dict N.graph_attr an attempt will be made first
    to copy properties attached to the graph (see from_agraph)
    and then updated with the calling arguments if any.

    """
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError("requires pygraphviz http://pygraphviz.github.io/") from err
    directed = N.is_directed()
    strict = eg.number_of_selfloops(N) == 0 and not N.is_multigraph()
    A = pygraphviz.AGraph(name=N.name, strict=strict, directed=directed)

    # default graph attributes
    A.graph_attr.update(N.graph.get("graph", {}))
    A.node_attr.update(N.graph.get("node", {}))
    A.edge_attr.update(N.graph.get("edge", {}))

    A.graph_attr.update(
        (k, v) for k, v in N.graph.items() if k not in ("graph", "node", "edge")
    )

    # add nodes
    for n, nodedata in N.nodes(data=True):
        A.add_node(n)
        # Add node data
        a = A.get_node(n)
        a.attr.update({k: str(v) for k, v in nodedata.items()})

    # loop over edges
    if N.is_multigraph():
        for u, v, key, edgedata in N.edges(data=True, keys=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items() if k != "key"}
            A.add_edge(u, v, key=str(key))
            # Add edge data
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)

    else:
        for u, v, edgedata in N.edges(data=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items()}
            A.add_edge(u, v)
            # Add edge data
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)

    return A
