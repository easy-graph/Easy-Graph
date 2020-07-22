__all__ = [
    "to_numpy_matrix"
]

def to_numpy_matrix(G, edge_sign = 1.0, not_edge_sign = 0.0):
    """
    Returns the graph adjacency matrix as a NumPy matrix.

    Parameters
    ----------
    edge_sign : float
        Sign for the position of matrix where there is an edge
    
    not_edge_sign : float
        Sign for the position of matrix where there is no edge

    """
    import numpy as np
    index_of_node = dict(zip(G.nodes, range(len(G))))
    N = len(G)
    M = np.full((N, N), not_edge_sign)

    for u, udict in G.adj.items():
        for v, data in udict.items():
            M[index_of_node[u], index_of_node[v]] = edge_sign

    M = np.asmatrix(M)
    return M
