__all__ = ["hyepergraph_degree_centrality"]


def hyepergraph_degree_centrality(G):
    """

    Parameters
    ----------
    G : eg.Hypergraph
        The target hypergraph

    Returns
    ----------
    degree centrality of each node in G : dict

    """
    res = {}
    node_list = G.v
    # Get hyperedge list
    edge_list = G.e[0]
    for node in node_list:
        res[node] = 0

    for e in edge_list:
        for n in e:
            res[n] += 1

    return res
