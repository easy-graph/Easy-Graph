import easygraph as eg

from easygraph.utils import *


__all__ = ["pagerank"]


@not_implemented_for("multigraph")
@hybrid("cpp_pagerank")
def pagerank(G, alpha=0.85):
    """
    Returns the PageRank value of each node in G.

    Parameters
    ----------
    G : graph
        Undirected graph will be considered as directed graph with two directed edges for each undirected edge.

    alpha : float
        The damping factor. Default is 0.85

    """
    import numpy as np

    if len(G) == 0:
        return {}
    M = google_matrix(G, alpha=alpha)

    # use numpy LAPACK solver
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)
    # eigenvector of largest eigenvalue is at ind, normalized
    largest = np.array(eigenvectors[:, ind]).flatten().real
    norm = float(largest.sum())
    return dict(zip(G, map(float, largest / norm)))


def google_matrix(G, alpha):
    import numpy as np

    M = eg.to_numpy_array(G)
    N = len(G)
    if N == 0:
        return M

    # Get dangling nodes(nodes with no out link)
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]
    dangling_weights = np.repeat(1.0 / N, N)
    for node in dangling_nodes:
        M[node] = dangling_weights

    M /= M.sum(axis=1)[:, np.newaxis]

    return alpha * M + (1 - alpha) * np.repeat(1.0 / N, N)
