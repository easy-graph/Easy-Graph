__all__ = ["hypergraph_degree_centrality"]


def hypergraph_degree_centrality(G):
    """

    Parameters
    ----------
    G : eg.Hypergraph
        The target hypergraph

    Returns
    ----------
    hyperdegree of each node in G : dict

    """
    node_list = G.v
    edge_list = G.e[0]

    res = {node: 0 for node in node_list}

    for e in edge_list:
        res.update({node: res[node] + 1 for node in e})
    # res = {}
    # node_list = G.v
    # # Get hyperedge list
    # edge_list = G.e[0]
    # for node in node_list:
    #     res[node] = 0

    # for e in edge_list:
    #     for n in e:
    #         res[n] += 1

    return res
