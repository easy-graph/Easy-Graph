from easygraph.utils import *
import numpy as np
from easygraph.utils.decorators import *

__all__ = ["katz_centrality"]

@not_implemented_for("multigraph")
@hybrid("cpp_katz_centrality")
def katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-6, normalized=True):
    r"""
    Compute the Katz centrality for nodes in a graph.

    Katz centrality computes the influence of a node based on the total number 
    of walks between nodes, attenuated by a factor of their length. It is 
    defined as the solution to the linear system:

    .. math::

        x = \alpha A x + \beta

    where:
        - \( A \) is the adjacency matrix of the graph,
        - \( \alpha \) is a scalar attenuation factor,
        - \( \beta \) is the bias vector (typically all ones),
        - and \( x \) is the resulting centrality vector.

    The algorithm runs an iterative fixed-point method until convergence.

    Parameters
    ----------
    G : easygraph.Graph
        An EasyGraph graph instance. Must be simple (non-multigraph).

    alpha : float, optional (default=0.1)
        Attenuation factor, must be smaller than the reciprocal of the largest
        eigenvalue of the adjacency matrix to ensure convergence.

    beta : float or dict, optional (default=1.0)
        Bias term. Can be a constant scalar applied to all nodes, or a dictionary
        mapping node IDs to values.

    max_iter : int, optional (default=1000)
        Maximum number of iterations before the algorithm terminates.

    tol : float, optional (default=1e-6)
        Convergence tolerance. Iteration stops when the L1 norm of the difference
        between successive iterations is below this threshold.

    normalized : bool, optional (default=True)
        If True, the result vector will be normalized to unit norm (L2).

    Returns
    -------
    dict
        A dictionary mapping node IDs to Katz centrality scores.

    Raises
    ------
    RuntimeError
        If the algorithm fails to converge within `max_iter` iterations.

    Examples
    --------
    >>> import easygraph as eg
    >>> from easygraph import katz_centrality
    >>> G = eg.Graph()
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    >>> katz_centrality(G, alpha=0.05)
    {0: 0.370..., 1: 0.447..., 2: 0.447..., 3: 0.370...}
    """
    # Create node ordering
    nodes = list(G.nodes)
    n = len(nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    index_to_node = {i: node for i, node in enumerate(nodes)}

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.float64)
    for u in G.nodes:
        for v in G.adj[u]:
            A[node_to_index[u], node_to_index[v]] = 1.0

    # Initialize x and beta
    x = np.ones(n, dtype=np.float64)
    if isinstance(beta, dict):
        b = np.array([beta.get(index_to_node[i], 1.0) for i in range(n)])
    else:
        b = np.ones(n, dtype=np.float64) * beta

    # Iterative update using vectorized ops
    for _ in range(max_iter):
        x_new = alpha * A @ x + b
        if np.linalg.norm(x_new - x, ord=1) < tol:
            break
        x = x_new
    else:
        raise RuntimeError(f"Katz centrality failed to converge in {max_iter} iterations")

    if normalized:
        norm = np.linalg.norm(x)
        if norm > 0:
            x /= norm

    result = {index_to_node[i]: float(x[i]) for i in range(n)}
    return result
