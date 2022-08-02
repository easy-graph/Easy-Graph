from easygraph.utils import *


__all__ = ["laplacian"]


@not_implemented_for("multigraph")
def laplacian(G, n_workers=None):
    """Returns the laplacian centrality of each node in the weighted graph

    Parameters
    ----------
    G : graph
        weighted graph

    Returns
    -------
    CL : dict
        the laplacian centrality of each node in the weighted graph

    Examples
    --------
    Returns the laplacian centrality of each node in the weighted graph G

    >>> laplacian(G)

    Reference
    ---------
    .. [1] Xingqin Qi, Eddie Fuller, Qin Wu, Yezhou Wu, Cun-Quan Zhang.
    "Laplacian centrality: A new centrality measure for weighted networks."
    Information Sciences, Volume 194, Pages 240-253, 2012.

    """
    adj = G.adj
    from collections import defaultdict

    X = defaultdict(int)
    W = defaultdict(int)
    CL = {}

    if n_workers is not None:
        # use the parallel version for large graph
        import random

        from functools import partial
        from multiprocessing import Pool

        nodes = list(G.nodes)
        random.shuffle(nodes)

        if len(nodes) > n_workers * 30000:
            nodes = split_len(nodes, step=30000)
        else:
            nodes = split(nodes, n_workers)

        local_function = partial(initialize_parallel, G=G, adj=adj)
        with Pool(n_workers) as p:
            ret = p.imap(local_function, nodes)
            resX, resW = [], []
            for i in ret:
                for x in i:
                    resX.append(x[0])
                    resW.append(x[1])
            X = dict(resX)
            W = dict(resW)
            ELG = sum(X[i] * X[i] for i in G) + sum(W[i] for i in G)
        local_function = partial(laplacian_parallel, G=G, X=X, W=W, adj=adj, ELG=ELG)
        with Pool(n_workers) as p:
            ret = p.imap(local_function, nodes)
            res = [x for i in ret for x in i]
        CL = dict(res)

    else:
        # use np-parallel version for small graph
        for i in G:
            for j in G:
                if i in G and j in G[i]:
                    X[i] += adj[i][j].get("weight", 1)
                    W[i] += adj[i][j].get("weight", 1) * adj[i][j].get("weight", 1)
        ELG = sum(X[i] * X[i] for i in G) + sum(W[i] for i in G)
        for i in G:
            import copy

            Xi = copy.deepcopy(X)
            for j in G:
                if j in adj.keys() and i in adj[j].keys():
                    Xi[j] -= adj[j][i].get("weight", 1)
            Xi[i] = 0
            ELGi = sum(Xi[i] * Xi[i] for i in G) + sum(W[i] for i in G) - 2 * W[i]
            if ELG:
                CL[i] = (float)(ELG - ELGi) / ELG
    return CL


def initialize_parallel(nodes, G, adj):
    ret = []
    for i in nodes:
        X = 0
        W = 0
        for j in G:
            if j in G[i]:
                X += adj[i][j].get("weight", 1)
                W += adj[i][j].get("weight", 1) * adj[i][j].get("weight", 1)
        ret.append([[i, X], [i, W]])
    return ret


def laplacian_parallel(nodes, G, X, W, adj, ELG):
    ret = []
    for i in nodes:
        import copy

        Xi = copy.deepcopy(X)
        for j in G:
            if j in adj.keys() and i in adj[j].keys():
                Xi[j] -= adj[j][i].get("weight", 1)
        Xi[i] = 0
        ELGi = sum(Xi[i] * Xi[i] for i in G) + sum(W[i] for i in G) - 2 * W[i]
        if ELG:
            ret.append([i, (float)(ELG - ELGi) / ELG])
    return ret


def sort(data):
    return dict(sorted(data.items(), key=lambda x: x[0], reverse=True))


def output(data, path):
    import json

    data = sort(data)
    json_str = json.dumps(data, ensure_ascii=False, indent=4)
    with open(path, "w", encoding="utf-8") as json_file:
        json_file.write(json_str)
