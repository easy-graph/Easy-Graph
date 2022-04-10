import easygraph as eg

__all__ = [
    "parse_edgelist",
]


def parse_edgelist(lines,
                   comments="#",
                   delimiter=None,
                   create_using=None,
                   nodetype=None,
                   data=True):
    """Parse lines of an edge list representation of a graph.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in edgelist format
    comments : string, optional
       Marker for comment lines. Default is `'#'`. To specify that no character
       should be treated as a comment, use ``comments=None``.
    delimiter : string, optional
       Separator for node labels. Default is `None`, meaning any whitespace.
    create_using : EasyGraph graph constructor, optional (default=eg.Graph)
       Graph type to create. If graph instance, then cleared before populated.
    nodetype : Python type, optional
       Convert nodes to this type. Default is `None`, meaning no conversion is
       performed.
    data : bool or list of (label,type) tuples
       If `False` generate no edge data or if `True` use a dictionary
       representation of edge data or a list tuples specifying dictionary
       key names and types for edge data.

    Returns
    -------
    G: EasyGraph Graph
        The graph corresponding to lines

    Examples
    --------
    Edgelist with no data:

    >>> lines = ["1 2", "2 3", "3 4"]
    >>> G = eg.parse_edgelist(lines, nodetype=int)
    >>> list(G)
    [1, 2, 3, 4]
    >>> list(G.edges())
    [(1, 2), (2, 3), (3, 4)]

    Edgelist with data in Python dictionary representation:

    >>> lines = ["1 2 {'weight': 3}", "2 3 {'weight': 27}", "3 4 {'weight': 3.0}"]
    >>> G = eg.parse_edgelist(lines, nodetype=int)
    >>> list(G)
    [1, 2, 3, 4]
    >>> list(G.edges(data=True))
    [(1, 2, {'weight': 3}), (2, 3, {'weight': 27}), (3, 4, {'weight': 3.0})]

    Edgelist with data in a list:

    >>> lines = ["1 2 3", "2 3 27", "3 4 3.0"]
    >>> G = eg.parse_edgelist(lines, nodetype=int, data=(("weight", float),))
    >>> list(G)
    [1, 2, 3, 4]
    >>> list(G.edges(data=True))
    [(1, 2, {'weight': 3.0}), (2, 3, {'weight': 27.0}), (3, 4, {'weight': 3.0})]

    See Also
    --------
    read_weighted_edgelist
    """
    from ast import literal_eval

    G = eg.empty_graph(0, create_using)
    for line in lines:
        if comments is not None:
            p = line.find(comments)
            if p >= 0:
                line = line[:p]
            if not line:
                continue
        # split line, should have 2 or more
        s = line.strip().split(delimiter)
        if len(s) < 2:
            continue
        u = s.pop(0)
        v = s.pop(0)
        d = s
        if nodetype is not None:
            try:
                u = nodetype(u)
                v = nodetype(v)
            except Exception as err:
                raise TypeError(
                    f"Failed to convert nodes {u},{v} to type {nodetype}."
                ) from err

        if len(d) == 0 or data is False:
            # no data or data type specified
            edgedata = {}
        elif data is True:
            # no edge types specified
            try:  # try to evaluate as dictionary
                if delimiter == ",":
                    edgedata_str = ",".join(d)
                else:
                    edgedata_str = " ".join(d)
                edgedata = dict(literal_eval(edgedata_str.strip()))
            except Exception as err:
                raise TypeError(
                    f"Failed to convert edge data ({d}) to dictionary."
                ) from err
        else:
            # convert edge data to dictionary with specified keys and type
            if len(d) != len(data):
                raise IndexError(
                    f"Edge data {d} and data_keys {data} are not the same length"
                )
            edgedata = {}
            for (edge_key, edge_type), edge_value in zip(data, d):
                try:
                    edge_value = edge_type(edge_value)
                except Exception as err:
                    raise TypeError(
                        f"Failed to convert {edge_key} data {edge_value} "
                        f"to type {edge_type}.") from err
                edgedata.update({edge_key: edge_value})
        G.add_edge(u, v, **edgedata)
    return G
