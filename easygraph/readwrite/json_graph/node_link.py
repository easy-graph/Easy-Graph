from itertools import chain
from itertools import count

import easygraph as eg


__all__ = ["node_link_graph"]


_attrs = dict(source="source", target="target", name="id", key="key", link="links")


def _to_tuple(x):
    """Converts lists to tuples, including nested lists.

    All other non-list inputs are passed through unmodified. This function is
    intended to be used to convert potentially nested lists from json files
    into valid nodes.

    Examples
    --------
    >>> _to_tuple([1, 2, [3, 4]])
    (1, 2, (3, 4))
    """
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(map(_to_tuple, x))


def node_link_graph(data, directed=False, multigraph=True, attrs=None):
    """Returns graph from node-link data format.

    Parameters
    ----------
    data : dict
        node-link formatted graph data

    directed : bool
        If True, and direction not specified in data, return a directed graph.

    multigraph : bool
        If True, and multigraph not specified in data, return a multigraph.

    attrs : dict
        A dictionary that contains five keys 'source', 'target', 'name',
        'key' and 'link'.  The corresponding values provide the attribute
        names for storing NetworkX-internal graph data.  Default value:

            dict(source='source', target='target', name='id',
                key='key', link='links')

    Returns
    -------
    G : EasyGraph graph
        A EasyGraph graph object

    Examples
    --------
    >>> from easygraph.readwrite import json_graph
    >>> G = eg.Graph([("A", "B")])
    >>> data = json_graph.node_link_data(G)
    >>> H = json_graph.node_link_graph(data)

    Notes
    -----
    Attribute 'key' is only used for multigraphs.

    See Also
    --------
    node_link_data, adjacency_data, tree_data
    """
    # Allow 'attrs' to keep default values.
    if attrs is None:
        attrs = _attrs
    else:
        attrs.update({k: v for k, v in _attrs.items() if k not in attrs})
    multigraph = data.get("multigraph", multigraph)
    directed = data.get("directed", directed)
    if multigraph:
        graph = eg.MultiGraph()
    else:
        graph = eg.Graph()
    if directed:
        graph = graph.to_directed()
    name = attrs["name"]
    source = attrs["source"]
    target = attrs["target"]
    links = attrs["link"]
    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else attrs["key"]
    graph.graph = data.get("graph", {})
    c = count()
    for d in data["nodes"]:
        node = _to_tuple(d.get(name, next(c)))
        nodedata = {str(k): v for k, v in d.items() if k != name}
        graph.add_node(node, **nodedata)
    for d in data[links]:
        src = tuple(d[source]) if isinstance(d[source], list) else d[source]
        tgt = tuple(d[target]) if isinstance(d[target], list) else d[target]
        if not multigraph:
            edgedata = {str(k): v for k, v in d.items() if k != source and k != target}
            graph.add_edge(src, tgt, **edgedata)
        else:
            ky = d.get(key, None)
            edgedata = {
                str(k): v
                for k, v in d.items()
                if k != source and k != target and k != key
            }
            graph.add_edge(src, tgt, ky, **edgedata)
    return graph
