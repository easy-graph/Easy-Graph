import easygraph as eg


__all__ = ["relabel_nodes", "convert_node_labels_to_integers"]


def relabel_nodes(G, mapping):
    if not hasattr(mapping, "__getitem__"):
        m = {n: mapping(n) for n in G}
    else:
        m = mapping
    return _relabel_copy(G, m)


def _relabel_copy(G, mapping):
    H = G.__class__()
    H.add_nodes_from(mapping.get(n, n) for n in G)
    H._node.update((mapping.get(n, n), d.copy()) for n, d in G.nodes.items())
    if G.is_multigraph():
        new_edges = [
            (mapping.get(n1, n1), mapping.get(n2, n2), k, d.copy())
            for (n1, n2, k, d) in G.edges
        ]

        # check for conflicting edge-keys
        undirected = not G.is_directed()
        seen_edges = set()
        for i, (source, target, key, data) in enumerate(new_edges):
            while (source, target, key) in seen_edges:
                if not isinstance(key, (int, float)):
                    key = 0
                key += 1
            seen_edges.add((source, target, key))
            if undirected:
                seen_edges.add((target, source, key))
            new_edges[i] = (source, target, key, data)

        H.add_edges_from(new_edges)
    else:
        H.add_edges_from(
            (mapping.get(n1, n1), mapping.get(n2, n2), d.copy())
            for (n1, n2, d) in G.edges
        )
    H.graph.update(G.graph)
    return H


def convert_node_labels_to_integers(
    G, first_label=0, ordering="default", label_attribute=None
):
    """Returns a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A easygraph graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    ordering : string
       "default" : inherit node ordering from G.nodes
       "sorted"  : inherit node ordering from sorted(G.nodes)
       "increasing degree" : nodes are sorted by increasing degree
       "decreasing degree" : nodes are sorted by decreasing degree

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    There is no guarantee that the relabeling of nodes to integers will
    give the same two integers for two (even identical graphs).
    Use the `ordering` argument to try to preserve the order.

    See Also
    --------
    relabel_nodes
    """
    N = G.number_of_nodes() + first_label
    if ordering == "default":
        mapping = dict(zip(G.nodes, range(first_label, N)))
    elif ordering == "sorted":
        nlist = sorted(G.nodes)
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == "increasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    elif ordering == "decreasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise eg.EasyGraphError(f"Unknown node ordering: {ordering}")
    H = relabel_nodes(G, mapping)
    # create node attribute with the old label
    if label_attribute is not None:
        eg.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
    return H
