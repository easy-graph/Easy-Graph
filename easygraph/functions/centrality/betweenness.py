from easygraph.utils import *
from easygraph.utils.decorators import *


__all__ = [
    "betweenness_centrality",
]


def betweenness_centrality_parallel(nodes, G, path_length, accumulate):
    betweenness = {node: 0.0 for node in G}
    for node in nodes:
        S, P, sigma = path_length(G, source=node)
        betweenness = accumulate(betweenness, S, P, sigma, node)
    return betweenness


@not_implemented_for("multigraph")
@hybrid("cpp_betweenness_centrality")
def betweenness_centrality(
    G, weight=None, sources=None, normalized=True, endpoints=False, n_workers=None
):
    r"""Compute the shortest-basic betweenness centrality for nodes.

    .. math::

        c_B(v)  = \sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where V is the set of nodes,

    .. math::
        \sigma(s, t)

    is the number of shortest (s, t)-paths, and

    .. math::

        \sigma(s, t|v)

    is the number of those paths  passing through some node v other than s, t.

    .. math::

        If\ s\ =\ t,\ \sigma(s, t) = 1, and\ if\ v \in {s, t}, \sigma(s, t|v) = 0 [2]_.

    Parameters
    ----------
    G : graph
      A easygraph graph.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    sources : None or nodes list, optional (default=None)
      If None, all nodes are considered.
      Otherwise,the set of source vertices to consider when calculating shortest paths.

    normalized : bool, optional
      If True the betweenness values are normalized by `2/((n-1)(n-2))`
      for graphs, and `1/((n-1)(n-2))` for directed graphs where `n`
      is the number of nodes in G.

    endpoints : bool, optional
      If True include the endpoints in the shortest basic counts.

    Returns
    -------

    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    >>> betweenness_centrality(G,weight="weight")
    """

    import functools

    if weight is not None:
        path_length = functools.partial(_single_source_dijkstra_path, weight=weight)
    else:
        path_length = functools.partial(_single_source_bfs_path)

    if endpoints:
        accumulate = functools.partial(_accumulate_endpoints)
    else:
        accumulate = functools.partial(_accumulate_basic)

    if sources is not None:
        nodes = sources
    else:
        nodes = G.nodes
    betweenness = dict.fromkeys(G, 0.0)

    if n_workers is not None:
        #  use the parallel version for large graph
        import random

        from functools import partial
        from multiprocessing import Pool

        nodes = list(nodes)
        random.shuffle(nodes)

        if len(nodes) > n_workers * 30000:
            nodes = split_len(nodes, step=30000)
        else:
            nodes = split(nodes, n_workers)
        local_function = partial(
            betweenness_centrality_parallel,
            G=G,
            path_length=path_length,
            accumulate=accumulate,
        )
        with Pool(n_workers) as p:
            ret = p.imap(local_function, nodes)
            for res in ret:
                for key in res:
                    betweenness[key] += res[key]
    else:
        # use np-parallel version for small graph
        for node in nodes:
            S, P, sigma = path_length(G, source=node)
            betweenness = accumulate(betweenness, S, P, sigma, node)

    betweenness = _rescale(
        betweenness,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        endpoints=endpoints,
    )
    ret = [0.0 for i in range(len(G))]
    for i in range(len(ret)):
        ret[i] = betweenness[G.index2node[i]]
    return ret


def _rescale(betweenness, n, normalized, directed=False, endpoints=False):
    if normalized:
        if endpoints:
            if n < 2:
                scale = None  # no normalization
            else:
                # Scale factor should include endpoint nodes
                scale = 1 / (n * (n - 1))
        elif n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def _single_source_bfs_path(G, source):
    S = []
    P = {v: [] for v in G}
    sigma = dict.fromkeys(G, 0.0)
    D = {}
    sigma[source] = 1.0
    D[source] = 0
    Q = [source]
    adj = G.adj
    while Q:
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in adj[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                sigma[w] += sigmav
                P[w].append(v)
    return S, P, sigma


def _single_source_dijkstra_path(G, source, weight="weight"):
    from heapq import heappop
    from heapq import heappush

    push = heappush
    pop = heappop
    S = []
    P = {v: [] for v in G}
    sigma = dict.fromkeys(G, 0.0)
    D = {}
    sigma[source] = 1.0
    seen = {source: 0}
    Q = []
    from itertools import count

    c = count()
    adj = G.adj
    push(Q, (0, next(c), source, source))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue
        sigma[v] += sigma[pred]
        S.append(v)
        D[v] = dist
        for w in adj[v]:
            vw_dist = dist + adj[v][w].get(weight, 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma


def _accumulate_endpoints(betweenness, S, P, sigma, s):
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return betweenness


def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness
