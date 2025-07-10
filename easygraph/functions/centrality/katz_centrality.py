from easygraph.utils.decorators import not_implemented_for
from easygraph.utils import *

__all__ = ["katz_centrality"]

@not_implemented_for("multigraph")
def katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-6, normalized=True):
    """Compute the Katz centrality of a graph.

    Parameters
    ----------
    G : graph
        A EasyGraph graph.
    alpha : float
        Attenuation factor (should be < 1 / largest eigenvalue).
    beta : float or dict
        Initial centrality (can be scalar or dict of node->value).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Error tolerance used to check convergence.
    normalized : bool
        Whether to normalize the resulting centralities.

    Returns
    -------
    centrality : dict
        Dictionary of nodes with Katz centrality values.
    """
    from collections import defaultdict

    nodes = list(G.nodes)
    A = G.adj
    centrality = {v: 1.0 for v in nodes}
    beta_vec = {v: beta if isinstance(beta, (int, float)) else beta.get(v, 1.0) for v in nodes}

    for i in range(max_iter):
        new_centrality = defaultdict(float)
        for v in nodes:
            for u in A[v]:
                new_centrality[v] += centrality[u]
        for v in nodes:
            new_centrality[v] = alpha * new_centrality[v] + beta_vec[v]

        # Check convergence
        err = sum(abs(new_centrality[v] - centrality[v]) for v in nodes)
        centrality = new_centrality
        if err < tol:
            break
    else:
        raise RuntimeError(f"Katz centrality failed to converge in {max_iter} iterations")

    if normalized:
        norm = sum(v**2 for v in centrality.values()) ** 0.5
        for v in centrality:
            centrality[v] /= norm

    return centrality
