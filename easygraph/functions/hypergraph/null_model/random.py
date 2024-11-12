import math
import random
import warnings

from collections import defaultdict
from itertools import combinations

import easygraph as eg
import numpy as np

# from easygraph.classes.hypergraph import Hypergraph
from easygraph.utils.exception import EasyGraphError
from scipy.special import comb

from .lattice import *


__all__ = [
    "random_hypergraph",
    "chung_lu_hypergraph",
    "dcsbm_hypergraph",
    "watts_strogatz_hypergraph",
    "uniform_hypergraph_Gnp",
]


def uniform_hypergraph_Gnp_parallel(edges, prob):
    remain_edges = [e for e in edges if random.random() < prob]
    return remain_edges


def split_edges(edges, worker):
    import math

    edges_size = len(edges)
    group_size = math.ceil(edges_size / worker)
    group_lst = []
    for i in range(0, edges_size, group_size):
        group_lst.append(edges[i : i + group_size])

    return group_lst


def uniform_hypergraph_Gnp(k: int, num_v: int, prob: float, n_workers=None):
    r"""Return a random ``k``-uniform hypergraph with ``num_v`` vertices and probability ``prob`` of choosing a hyperedge.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``k`` (``int``): The Number of vertices in each hyperedge.
        ``prob`` (``float``): Probability of choosing a hyperedge.

    Examples:
        >>> import easygraph as eg
        >>> hg = eg.random.uniform_hypergraph_Gnp(3, 5, 0.5)
        >>> hg.e
        ([(0, 1, 3), (0, 1, 4), (0, 2, 4), (1, 3, 4), (2, 3, 4)], [1.0, 1.0, 1.0, 1.0, 1.0])
    """
    # similar to BinomialRandomUniform in sagemath, https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hypergraph_generators.html

    assert num_v > 1, "num_v must be greater than 1"
    assert k > 1, "k must be greater than 1"
    assert 0 <= prob <= 1, "prob must be between 0 and 1"
    import random

    if n_workers is not None:
        #  use the parallel version for large graph

        from functools import partial
        from multiprocessing import Pool

        edges = combinations(range(num_v), k)
        edges_parallel = split_edges(edges=list(edges), worker=n_workers)
        local_function = partial(uniform_hypergraph_Gnp_parallel, prob=prob)

        res_edges = []

        with Pool(n_workers) as p:
            ret = p.imap(local_function, edges_parallel)
            for res in ret:
                res_edges.extend(res)
        res_hypergraph = eg.Hypergraph(num_v=num_v, e_list=res_edges)
        return res_hypergraph

    else:
        edges = combinations(range(num_v), k)
        edges = [e for e in edges if random.random() < prob]
        return eg.Hypergraph(num_v=num_v, e_list=edges)


def dcsbm_hypergraph(k1, k2, g1, g2, omega, seed=None):
    """A function to generate a Degree-Corrected Stochastic Block Model
    (DCSBM) hypergraph.

    Parameters
    ----------
    k1 : dict
        This is a dictionary where the keys are node ids
        and the values are node degrees.
    k2 : dict
        This is a dictionary where the keys are edge ids
        and the values are edge sizes.
    g1 : dict
        This a dictionary where the keys are node ids
        and the values are the group ids to which the node belongs.
        The keys must match the keys of k1.
    g2 : dict
        This a dictionary where the keys are edge ids
        and the values are the group ids to which the edge belongs.
        The keys must match the keys of k2.
    omega : 2D numpy array
        This is a matrix with entries which specify the number of edges
        between a given node community and edge community.
        The number of rows must match the number of node communities
        and the number of columns must match the number of edge
        communities.
    seed : int or None (default)
        Seed for the random number generator.

    Returns
    -------
    Hypergraph

    Warns
    -----
    warnings.warn
        If the sums of the edge sizes and node degrees are not equal, the
        algorithm still runs, but raises a warning.
        Also if the sum of the omega matrix does not match the sum of degrees,
        a warning is raised.

    Notes
    -----
    The sums of k1 and k2 should be the same. If they are not the same, this function
    returns a warning but still runs. The sum of k1 (and k2) and omega should be the
    same. If they are not the same, this function returns a warning but still runs and
    the number of entries in the incidence matrix is determined by the omega matrix.

    References
    ----------
    Implemented by Mirah Shi in HyperNetX and described for bipartite networks by
    Larremore et al. in https://doi.org/10.1103/PhysRevE.90.012805

    Examples
    --------
    >>> import easygraph as eg; import random; import numpy as np
    >>> n = 50
    >>> k1 = {i : random.randint(1, n) for i in range(n)}
    >>> k2 = {i : sorted(k1.values())[i] for i in range(n)}
    >>> g1 = {i : random.choice([0, 1]) for i in range(n)}
    >>> g2 = {i : random.choice([0, 1]) for i in range(n)}
    >>> omega = np.array([[n//2, 10], [10, n//2]])
    >>> H = eg.dcsbm_hypergraph(k1, k2, g1, g2, omega)

    """
    if seed is not None:
        random.seed(seed)

    # sort dictionary by degree in decreasing order
    node_labels = [n for n, _ in sorted(k1.items(), key=lambda d: d[1], reverse=True)]
    edge_labels = [m for m, _ in sorted(k2.items(), key=lambda d: d[1], reverse=True)]

    # Verify that the sum of node and edge degrees and the sum of node degrees and the
    # sum of community connection matrix differ by less than a single edge.
    if abs(sum(k1.values()) - sum(k2.values())) > 1:
        warnings.warn(
            "The sum of the degree sequence does not match the sum of the size sequence"
        )

    if abs(sum(k1.values()) - np.sum(omega)) > 1:
        warnings.warn(
            "The sum of the degree sequence does not "
            "match the entries in the omega matrix"
        )

    # get indices for each community
    community1_nodes = defaultdict(list)
    for label in node_labels:
        group = g1[label]
        community1_nodes[group].append(label)

    community2_nodes = defaultdict(list)
    for label in edge_labels:
        group = g2[label]
        community2_nodes[group].append(label)

    H = eg.Hypergraph(num_v=len(node_labels))

    kappa1 = defaultdict(lambda: 0)
    kappa2 = defaultdict(lambda: 0)
    for id, g in g1.items():
        kappa1[g] += k1[id]
    for id, g in g2.items():
        kappa2[g] += k2[id]

    tmp_hyperedges = []
    for group1 in community1_nodes.keys():
        for group2 in community2_nodes.keys():
            # for each constant probability patch
            try:
                group_constant = omega[group1, group2] / (
                    kappa1[group1] * kappa2[group2]
                )
            except ZeroDivisionError:
                group_constant = 0

            for u in community1_nodes[group1]:
                j = 0
                v = community2_nodes[group2][j]  # start from beginning every time
                # max probability
                p = min(k1[u] * k2[v] * group_constant, 1)
                while j < len(community2_nodes[group2]):
                    if p != 1:
                        r = random.random()
                        try:
                            j = j + math.floor(math.log(r) / math.log(1 - p))
                        except ZeroDivisionError:
                            j = np.inf
                    if j < len(community2_nodes[group2]):
                        v = community2_nodes[group2][j]
                        q = min((k1[u] * k2[v]) * group_constant, 1)
                        r = random.random()
                        if r < q / p:
                            # no duplicates
                            if v < len(tmp_hyperedges):
                                if u not in tmp_hyperedges[v]:
                                    tmp_hyperedges[v].append(u)
                            else:
                                tmp_hyperedges.append([u])

                        p = q
                        j = j + 1

    H.add_hyperedges(tmp_hyperedges)
    return H


def watts_strogatz_hypergraph(n, d, k, l, p, seed=None):
    """

    Parameters
    ----------
    n : int
    The number of nodes
    d : int
        Edge size
    k: int
        Number of edges of which a node is a part. Should be a multiple of 2.
    l: int
        Overlap between edges
    p  : float
        The probability of rewiring each edge
    seed

    Returns
    -------

    """
    if seed is not None:
        np.random.seed(seed)
    H = ring_lattice(n, d, k, l)
    to_remove = []
    to_add = []
    H_edges = H.e[0]
    for e in H_edges:
        if np.random.random() < p:
            to_remove.append(e)
            node = min(e)
            neighbors = np.random.choice(H.v, size=d - 1)
            to_add.append(np.append(neighbors, node))

    for e in to_remove:
        if e in H_edges:
            H_edges.remove(e)

    for e in to_add:
        H_edges.append(e)

    H = eg.Hypergraph(num_v=n, e_list=H_edges)
    # H.remove_hyperedges(to_remove)
    # print("watts_strogatz:",H.e)
    # H.add_hyperedges(to_add)

    return H


def chung_lu_hypergraph(k1, k2, seed=None):
    """A function to generate a Chung-Lu hypergraph

    Parameters
    ----------
    k1 : dict
        Dict where the keys are node ids
        and the values are node degrees.
    k2 : dict
        dict where the keys are edge ids
        and the values are edge sizes.
    seed : integer or None (default)
            The seed for the random number generator.

    Returns
    -------
    Hypergraph object
        The generated hypergraph

    Warns
    -----
    warnings.warn
        If the sums of the edge sizes and node degrees are not equal, the
        algorithm still runs, but raises a warning.

    Notes
    -----
    The sums of k1 and k2 should be the same. If they are not the same,
    this function returns a warning but still runs.

    References
    ----------
    Implemented by Mirah Shi in HyperNetX and described for
    bipartite networks by Aksoy et al. in https://doi.org/10.1093/comnet/cnx001

    Example
    -------
    >>> import easygraph as eg
    >>> import random
    >>> n = 100
    >>> k1 = {i : random.randint(1, 100) for i in range(n)}
    >>> k2 = {i : sorted(k1.values())[i] for i in range(n)}
    >>> H = eg.chung_lu_hypergraph(k1, k2)

    """
    if seed is not None:
        random.seed(seed)

    # sort dictionary by degree in decreasing order
    node_labels = [n for n, _ in sorted(k1.items(), key=lambda d: d[1], reverse=True)]
    edge_labels = [m for m, _ in sorted(k2.items(), key=lambda d: d[1], reverse=True)]

    m = len(k2)

    if sum(k1.values()) != sum(k2.values()):
        warnings.warn(
            "The sum of the degree sequence does not match the sum of the size sequence"
        )

    S = sum(k1.values())

    H = eg.Hypergraph(len(node_labels))

    tmp_hyperedges = []
    for u in node_labels:
        j = 0
        v = edge_labels[j]  # start from beginning every time
        p = min((k1[u] * k2[v]) / S, 1)

        while j < m:
            if p != 1:
                r = random.random()
                try:
                    j = j + math.floor(math.log(r) / math.log(1 - p))
                except ZeroDivisionError:
                    j = np.inf

            if j < m:
                v = edge_labels[j]
                q = min((k1[u] * k2[v]) / S, 1)
                r = random.random()
                if r < q / p:
                    # no duplicates
                    if v < len(tmp_hyperedges):
                        tmp_hyperedges[v].append(u)
                    else:
                        tmp_hyperedges.append([u])
                p = q
                j = j + 1

    H.add_hyperedges(tmp_hyperedges)
    return H


def random_hypergraph(N, ps, order=None, seed=None):
    """Generates a random hypergraph

    Generate N nodes, and connect any d+1 nodes
    by a hyperedge with probability ps[d-1].

    Parameters
    ----------
    N : int
        Number of nodes
    ps : list of float
        List of probabilities (between 0 and 1) to create a
        hyperedge at each order d between any d+1 nodes. For example,
        ps[0] is the wiring probability of any edge (2 nodes), ps[1]
        of any triangles (3 nodes).
    order: int of None (default)
        If None, ignore. If int, generates a uniform hypergraph with edges
        of order `order` (ps must have only one element).
    seed : integer or None (default)
            Seed for the random number generator.

    Returns
    -------
    Hypergraph object
        The generated hypergraph

    References
    ----------
    Described as 'random hypergraph' by M. Dewar et al. in https://arxiv.org/abs/1703.07686

    Example
    -------
    >>> import easygraph as eg
    >>> H = eg.random_hypergraph(50, [0.1, 0.01])

    """
    if seed is not None:
        np.random.seed(seed)

    if order is not None:
        if len(ps) != 1:
            raise EasyGraphError("ps must contain a single element if order is an int")

    if (np.any(np.array(ps) < 0)) or (np.any(np.array(ps) > 1)):
        raise EasyGraphError("All elements of ps must be between 0 and 1 included.")

    nodes = range(N)
    hyperedges = []

    for i, p in enumerate(ps):
        if order is not None:
            d = order
        else:
            d = i + 1  # order, ps[0] is prob of edges (d=1)

        potential_edges = combinations(nodes, d + 1)
        n_comb = comb(N, d + 1, exact=True)
        mask = np.random.random(size=n_comb) <= p  # True if edge to keep

        edges_to_add = [e for e, val in zip(potential_edges, mask) if val]

        hyperedges += edges_to_add

    H = eg.Hypergraph(num_v=N)
    H.add_hyperedges(hyperedges)

    return H
