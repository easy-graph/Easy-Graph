from collections import Counter
from itertools import chain

import numpy as np

from easygraph.utils.decorators import hybrid
from easygraph.utils.decorators import not_implemented_for
from easygraph.utils.misc import split
from easygraph.utils.misc import split_len


__all__ = ["average_clustering", "clustering"]


def _local_weighted_triangles_and_degree_iter_parallel(
    nodes_nbrs, G, weight, max_weight
):
    ret = []

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    for i, nbrs in nodes_nbrs:
        inbrs = set(nbrs) - {i}
        weighted_triangles = 0
        seen = set()
        for j in inbrs:
            seen.add(j)
            # This avoids counting twice -- we double at the end.
            jnbrs = set(G[j]) - seen
            # Only compute the edge weight once, before the inner inner
            # loop.
            wij = wt(i, j)
            weighted_triangles += sum(
                np.cbrt([(wij * wt(j, k) * wt(k, i)) for k in inbrs & jnbrs])
            )
        ret.append((i, len(inbrs), 2 * weighted_triangles))
    return ret


@not_implemented_for("multigraph")
def _weighted_triangles_and_degree_iter(G, nodes=None, weight="weight", n_workers=None):
    """Return an iterator of (node, degree, weighted_triangles).

    Used for weighted clustering.
    Note: this returns the geometric average weight of edges in the triangle.
    Also, each triangle is counted twice (each direction).
    So you may want to divide by 2.

    """

    if weight is None or G.number_of_edges() == 0:
        max_weight = 1
    else:
        max_weight = max(d.get(weight, 1) for u, v, d in G.edges)
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    if n_workers is not None:
        import random

        from functools import partial
        from multiprocessing import Pool

        _local_weighted_triangles_and_degree_iter_function = partial(
            _local_weighted_triangles_and_degree_iter_parallel,
            G=G,
            weight=weight,
            max_weight=max_weight,
        )
        nodes_nbrs = list(nodes_nbrs)
        random.shuffle(nodes_nbrs)
        if len(nodes_nbrs) > n_workers * 30000:
            nodes_nbrs = split_len(nodes, step=30000)
        else:
            nodes_nbrs = split(nodes_nbrs, n_workers)
        with Pool(n_workers) as p:
            ret = p.imap(_local_weighted_triangles_and_degree_iter_function, nodes_nbrs)
            for r in ret:
                for x in r:
                    yield x
    else:
        for i, nbrs in nodes_nbrs:
            inbrs = set(nbrs) - {i}
            weighted_triangles = 0
            seen = set()
            for j in inbrs:
                seen.add(j)
                # This avoids counting twice -- we double at the end.
                jnbrs = set(G[j]) - seen
                # Only compute the edge weight once, before the inner inner
                # loop.
                wij = wt(i, j)
                weighted_triangles += sum(
                    np.cbrt([(wij * wt(j, k) * wt(k, i)) for k in inbrs & jnbrs])
                )
            yield (i, len(inbrs), 2 * weighted_triangles)


def _local_directed_weighted_triangles_and_degree_parallel(
    nodes_nbrs, G, weight, max_weight
):
    ret = []

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    for i, preds, succs in nodes_nbrs:
        ipreds = set(preds) - {i}
        isuccs = set(succs) - {i}

        directed_triangles = 0
        for j in ipreds:
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._adj[j]) - {j}
            directed_triangles += sum(
                np.cbrt([(wt(j, i) * wt(k, i) * wt(k, j)) for k in ipreds & jpreds])
            )
            directed_triangles += sum(
                np.cbrt([(wt(j, i) * wt(k, i) * wt(j, k)) for k in ipreds & jsuccs])
            )
            directed_triangles += sum(
                np.cbrt([(wt(j, i) * wt(i, k) * wt(k, j)) for k in isuccs & jpreds])
            )
            directed_triangles += sum(
                np.cbrt([(wt(j, i) * wt(i, k) * wt(j, k)) for k in isuccs & jsuccs])
            )

        for j in isuccs:
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._adj[j]) - {j}
            directed_triangles += sum(
                np.cbrt([(wt(i, j) * wt(k, i) * wt(k, j)) for k in ipreds & jpreds])
            )
            directed_triangles += sum(
                np.cbrt([(wt(i, j) * wt(k, i) * wt(j, k)) for k in ipreds & jsuccs])
            )
            directed_triangles += sum(
                np.cbrt([(wt(i, j) * wt(i, k) * wt(k, j)) for k in isuccs & jpreds])
            )
            directed_triangles += sum(
                np.cbrt([(wt(i, j) * wt(i, k) * wt(j, k)) for k in isuccs & jsuccs])
            )

        dtotal = len(ipreds) + len(isuccs)
        dbidirectional = len(ipreds & isuccs)
        ret.append([i, dtotal, dbidirectional, directed_triangles])
    return ret


@not_implemented_for("multigraph")
def _directed_weighted_triangles_and_degree_iter(
    G, nodes=None, weight="weight", n_workers=None
):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_weighted_triangles).

    Used for directed weighted clustering.
    Note that unlike `_weighted_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """

    if weight is None or G.number_of_edges() == 0:
        max_weight = 1
    else:
        max_weight = max(d.get(weight, 1) for u, v, d in G.edges)

    nodes_nbrs = ((n, G._pred[n], G._adj[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    if n_workers is not None:
        import random

        from functools import partial
        from multiprocessing import Pool

        _local_directed_weighted_triangles_and_degree_function = partial(
            _local_directed_weighted_triangles_and_degree_parallel,
            G=G,
            weight=weight,
            max_weight=max_weight,
        )
        nodes_nbrs = list(nodes_nbrs)
        random.shuffle(nodes_nbrs)
        if len(nodes_nbrs) > n_workers * 30000:
            nodes_nbrs = split_len(nodes, step=30000)
        else:
            nodes_nbrs = split(nodes_nbrs, n_workers)
        with Pool(n_workers) as p:
            ret = p.imap(
                _local_directed_weighted_triangles_and_degree_function, nodes_nbrs
            )
            for r in ret:
                for x in r:
                    yield x

    else:
        for i, preds, succs in nodes_nbrs:
            ipreds = set(preds) - {i}
            isuccs = set(succs) - {i}

            directed_triangles = 0
            for j in ipreds:
                jpreds = set(G._pred[j]) - {j}
                jsuccs = set(G._adj[j]) - {j}
                directed_triangles += sum(
                    np.cbrt([(wt(j, i) * wt(k, i) * wt(k, j)) for k in ipreds & jpreds])
                )
                directed_triangles += sum(
                    np.cbrt([(wt(j, i) * wt(k, i) * wt(j, k)) for k in ipreds & jsuccs])
                )
                directed_triangles += sum(
                    np.cbrt([(wt(j, i) * wt(i, k) * wt(k, j)) for k in isuccs & jpreds])
                )
                directed_triangles += sum(
                    np.cbrt([(wt(j, i) * wt(i, k) * wt(j, k)) for k in isuccs & jsuccs])
                )

            for j in isuccs:
                jpreds = set(G._pred[j]) - {j}
                jsuccs = set(G._adj[j]) - {j}
                directed_triangles += sum(
                    np.cbrt([(wt(i, j) * wt(k, i) * wt(k, j)) for k in ipreds & jpreds])
                )
                directed_triangles += sum(
                    np.cbrt([(wt(i, j) * wt(k, i) * wt(j, k)) for k in ipreds & jsuccs])
                )
                directed_triangles += sum(
                    np.cbrt([(wt(i, j) * wt(i, k) * wt(k, j)) for k in isuccs & jpreds])
                )
                directed_triangles += sum(
                    np.cbrt([(wt(i, j) * wt(i, k) * wt(j, k)) for k in isuccs & jsuccs])
                )

            dtotal = len(ipreds) + len(isuccs)
            dbidirectional = len(ipreds & isuccs)
            yield (i, dtotal, dbidirectional, directed_triangles)


def average_clustering(G, nodes=None, weight=None, count_zeros=True, n_workers=None):
    r"""Compute the average clustering coefficient for the graph G.

    The clustering coefficient for the graph is the average,

    .. math::

       C = \frac{1}{n}\sum_{v \in G} c_v,

    where :math:`n` is the number of nodes in `G`.

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute average clustering for nodes in this container.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    count_zeros : bool
       If False include only the nodes with nonzero clustering in the average.

    Returns
    -------
    avg : float
       Average clustering

    Examples
    --------
    >>> G = eg.complete_graph(5)
    >>> print(eg.average_clustering(G))
    1.0

    Notes
    -----
    This is a space saving routine; it might be faster
    to use the clustering function to get a list and then take the average.

    Self loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Marcus Kaiser,  Mean clustering coefficients: the role of isolated
       nodes and leafs on clustering measures for small-world networks.
       https://arxiv.org/abs/0802.2512
    """
    c = clustering(G, nodes, weight=weight, n_workers=n_workers).values()
    if not count_zeros:
        c = [v for v in c if abs(v) > 0]
    return sum(c) / len(c)


def _local_directed_triangles_and_degree_iter_parallel(nodes_nbrs, G):
    ret = []
    for i, preds, succs in nodes_nbrs:
        ipreds = set(preds) - {i}
        isuccs = set(succs) - {i}

        directed_triangles = 0
        for j in chain(ipreds, isuccs):
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._adj[j]) - {j}
            directed_triangles += sum(
                1
                for k in chain(
                    (ipreds & jpreds),
                    (ipreds & jsuccs),
                    (isuccs & jpreds),
                    (isuccs & jsuccs),
                )
            )
        dtotal = len(ipreds) + len(isuccs)
        dbidirectional = len(ipreds & isuccs)
        ret.append((i, dtotal, dbidirectional, directed_triangles))
    return ret


@not_implemented_for("multigraph")
def _directed_triangles_and_degree_iter(G, nodes=None, n_workers=None):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_triangles).

    Used for directed clustering.
    Note that unlike `_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """
    nodes_nbrs = ((n, G._pred[n], G._adj[n]) for n in G.nbunch_iter(nodes))

    if n_workers is not None:
        import random

        from functools import partial
        from multiprocessing import Pool

        _local_directed_triangles_and_degree_iter_parallel_function = partial(
            _local_directed_triangles_and_degree_iter_parallel, G=G
        )
        nodes_nbrs = list(nodes_nbrs)
        random.shuffle(nodes_nbrs)
        if len(nodes_nbrs) > n_workers * 30000:
            nodes_nbrs = split_len(nodes_nbrs, step=30000)
        else:
            nodes_nbrs = split(nodes_nbrs, n_workers)

        with Pool(n_workers) as p:
            ret = p.imap(
                _local_directed_triangles_and_degree_iter_parallel_function, nodes_nbrs
            )
            for r in ret:
                for x in r:
                    yield x
    else:
        for i, preds, succs in nodes_nbrs:
            ipreds = set(preds) - {i}
            isuccs = set(succs) - {i}

            directed_triangles = 0
            for j in chain(ipreds, isuccs):
                jpreds = set(G._pred[j]) - {j}
                jsuccs = set(G._adj[j]) - {j}
                directed_triangles += sum(
                    1
                    for k in chain(
                        (ipreds & jpreds),
                        (ipreds & jsuccs),
                        (isuccs & jpreds),
                        (isuccs & jsuccs),
                    )
                )
            dtotal = len(ipreds) + len(isuccs)
            dbidirectional = len(ipreds & isuccs)
            yield (i, dtotal, dbidirectional, directed_triangles)


def _local_triangles_and_degree_iter_function_parallel(nodes_nbrs, G):
    ret = []
    for v, v_nbrs in nodes_nbrs:
        vs = set(v_nbrs) - {v}
        gen_degree = Counter(len(vs & (set(G[w]) - {w})) for w in vs)
        ntriangles = sum(k * val for k, val in gen_degree.items())
        ret.append((v, len(vs), ntriangles, gen_degree))
    return ret


@not_implemented_for("multigraph")
def _triangles_and_degree_iter(G, nodes=None, n_workers=None):
    """Return an iterator of (node, degree, triangles, generalized degree).

    This double counts triangles so you may want to divide by 2.
    See degree(), triangles() and generalized_degree() for definitions
    and details.

    """
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))

    if n_workers is not None:
        import random

        from functools import partial
        from multiprocessing import Pool

        _local_triangles_and_degree_iter_function = partial(
            _local_triangles_and_degree_iter_function_parallel, G=G
        )
        nodes_nbrs = list(nodes_nbrs)
        random.shuffle(nodes_nbrs)
        if len(nodes_nbrs) > n_workers * 30000:
            nodes_nbrs = split_len(nodes_nbrs, step=30000)
        else:
            nodes_nbrs = split(nodes_nbrs, n_workers)

        with Pool(n_workers) as p:
            ret = p.imap(_local_triangles_and_degree_iter_function, nodes_nbrs)
            for r in ret:
                for x in r:
                    yield x
    else:
        for v, v_nbrs in nodes_nbrs:
            vs = set(v_nbrs) - {v}
            gen_degree = Counter(len(vs & (set(G[w]) - {w})) for w in vs)
            ntriangles = sum(k * val for k, val in gen_degree.items())
            yield (v, len(vs), ntriangles, gen_degree)


@hybrid("cpp_clustering")
def clustering(G, nodes=None, weight=None, n_workers=None):
    r"""Compute the clustering coefficient for nodes.

    For unweighted graphs, the clustering of a node :math:`u`
    is the fraction of possible triangles through that node that exist,

    .. math::

      c_u = \frac{2 T(u)}{deg(u)(deg(u)-1)},

    where :math:`T(u)` is the number of triangles through node :math:`u` and
    :math:`deg(u)` is the degree of :math:`u`.

    For weighted graphs, there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2]_,

    .. math::

       c_u = \frac{1}{deg(u)(deg(u)-1))}
             \sum_{vw} (\hat{w}_{uv} \hat{w}_{uw} \hat{w}_{vw})^{1/3}.

    The edge weights :math:`\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\hat{w}_{uv} = w_{uv}/\max(w)`.

    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

    Additionally, this weighted definition has been generalized to support negative edge weights [3]_.

    For directed graphs, the clustering is similarly defined as the fraction
    of all possible directed triangles or geometric average of the subgraph
    edge weights for unweighted and weighted directed graph respectively [4]_.

    .. math::

       c_u = \frac{2}{deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\leftrightarrow}(u)}
             T(u),

    where :math:`T(u)` is the number of directed triangles through node
    :math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
    :math:`u` and :math:`deg^{\leftrightarrow}(u)` is the reciprocal degree of
    :math:`u`.


    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute clustering for nodes in this container.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    out : float, or dictionary
       Clustering coefficient at specified nodes

    Examples
    --------
    >>> G = eg.complete_graph(5)
    >>> print(eg.clustering(G, 0))
    1.0
    >>> print(eg.clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    Self loops are ignored.

        References
        ----------
        .. [1] Generalizations of the clustering coefficient to weighted
           complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
           K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
           http://jponnela.com/web_documents/a9.pdf
        .. [2] Intensity and coherence of motifs in weighted complex
           networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
           Physical Review E, 71(6), 065103 (2005).
        .. [3] Generalization of Clustering Coefficients to Signed Correlation Networks
           by G. Costantini and M. Perugini, PloS one, 9(2), e88669 (2014).
        .. [4] Clustering in complex directed networks by G. Fagiolo,
           Physical Review E, 76(2), 026107 (2007).
    """

    if G.is_directed():
        if weight is not None:
            td_iter = _directed_weighted_triangles_and_degree_iter(
                G, nodes, weight, n_workers=n_workers
            )
            clusterc = {
                v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2)
                for v, dt, db, t in td_iter
            }
        else:
            td_iter = _directed_triangles_and_degree_iter(G, nodes, n_workers=n_workers)
            clusterc = {
                v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2)
                for v, dt, db, t in td_iter
            }
    else:
        # The formula 2*T/(d*(d-1)) from docs is t/(d*(d-1)) here b/c t==2*T
        if weight is not None:
            td_iter = _weighted_triangles_and_degree_iter(
                G, nodes, weight, n_workers=n_workers
            )
            clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t in td_iter}
        else:
            td_iter = _triangles_and_degree_iter(G, nodes, n_workers=n_workers)
            clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t, _ in td_iter}
    if nodes in G:
        # Return the value of the sole entry in the dictionary.
        return clusterc[nodes]
    return clusterc
