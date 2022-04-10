import easygraph as eg

__all__ = ["relabel_nodes"]


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
        new_edges = [(mapping.get(n1, n1), mapping.get(n2, n2), k, d.copy())
                     for (n1, n2, k, d) in G.edges]

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
        H.add_edges_from((mapping.get(n1, n1), mapping.get(n2, n2), d.copy())
                         for (n1, n2, d) in G.edges)
    H.graph.update(G.graph)
    return H
