from easygraph.functions.basic import *
from easygraph.functions.path import single_source_bfs
from easygraph.functions.path import single_source_dijkstra
from easygraph.utils import *


__all__ = [
    "closeness_centrality",
]


def closeness_centrality_parallel(nodes, G, path_length):
    ret = []
    length = len(G)
    for node in nodes:
        x = path_length(G, node)
        dist = sum(x.values())
        cnt = len(x)
        if dist == 0:
            ret.append([node, 0])
        else:
            ret.append([node, (cnt - 1) * (cnt - 1) / (dist * (length - 1))])
    return ret


@not_implemented_for("multigraph")
@hybrid("cpp_closeness_centrality")
def closeness_centrality(G, weight=None, sources=None, n_workers=None):
    r"""
    Compute closeness centrality for nodes.

    .. math::

        C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    Notice that the closeness distance function computes the
    outcoming distance to `u` for directed graphs. To use
    incoming distance, act on `G.reverse()`.

    Parameters
    ----------
    G : graph
      A easygraph graph

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    sources : None or nodes list, optional (default=None)
      If None, all nodes are returned
      Otherwise,the set of source vertices to creturn.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.
    """
    closeness = dict()
    if sources is not None:
        nodes = sources
    else:
        nodes = G.nodes
    length = len(G)
    import functools

    if weight is not None:
        path_length = functools.partial(single_source_dijkstra, weight=weight)
    else:
        path_length = functools.partial(single_source_bfs)

    if n_workers is not None:
        # use parallel version for large graph
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
            closeness_centrality_parallel, G=G, path_length=path_length
        )
        with Pool(n_workers) as p:
            ret = p.imap(local_function, nodes)
            res = [x for i in ret for x in i]
        closeness = dict(res)
    else:
        # use np-parallel version for small graph
        for node in nodes:
            x = path_length(G, node)
            dist = sum(x.values())
            cnt = len(x)
            if dist == 0:
                closeness[node] = 0
            else:
                closeness[node] = (cnt - 1) * (cnt - 1) / (dist * (length - 1))
    ret = [0.0 for i in range(len(G))]
    for i in range(len(ret)):
        ret[i] = closeness[G.index2node[i]]
    return ret
