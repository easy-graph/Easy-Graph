
from .graph import DistributedGraph
from .result import DistributedResult


def pagerank(G, alpha=0.85, weight=None, *, max_iter=500, tol=1e-6):
    from .session import RemoteGraph
    if isinstance(G, RemoteGraph):
        return G.session.pagerank(G, alpha=alpha, weight=weight, max_iter=max_iter, tol=tol)
    if not isinstance(G, DistributedGraph):
        raise TypeError("Expected a DistributedGraph or DistributedDiGraph")
    ctx = G.context
    ctx._ensure_open()
    ctx._comm.check_error(
        "Only weight=None or weight='weight' is supported."
        if weight not in (None, "weight") else ""
    )
    ids, values, iterations = ctx._backend.pagerank(
        G._native, alpha, max_iter, tol, weight is not None
    )
    return DistributedResult(ctx, dict(zip(ids, values)), iterations=iterations)
