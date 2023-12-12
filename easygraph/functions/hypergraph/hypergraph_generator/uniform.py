"""Generate random uniform hypergraphs."""
import itertools
import operator
import random
import warnings

from functools import reduce

import easygraph as eg
import numpy as np

from easygraph.utils.exception import EasyGraphError


__all__ = [
    "uniform_hypergraph_configuration_model",
    "uniform_HSBM",
    "uniform_HPPM",
    "uniform_erdos_renyi_hypergraph",
    "uniform_hypergraph_Gnm",
]


def split_num_e(num_e, worker):
    import math

    res = []
    group_size = num_e // worker
    for i in range(worker):
        res.append(group_size)
    return res


def uniform_hypergraph_Gnm_parallel(num_e, num_v, k):
    random.seed()
    edges = set()
    while len(edges) < num_e:
        e = random.sample(range(num_v), k)
        e = tuple(sorted(e))
        if e not in edges:
            edges.add(e)
    return list(edges)


def uniform_hypergraph_Gnm(k: int, num_v: int, num_e: int, n_workers=None):
    r"""Return a random ``k``-uniform hypergraph with ``num_v`` vertices and ``num_e`` hyperedges.

    Args:
        ``k`` (``int``): The Number of vertices in each hyperedge.
        ``num_v`` (``int``): The Number of vertices.
        ``num_e`` (``int``): The Number of hyperedges.

    Examples:
        >>> import easygraph as eg
        >>> hg = eg.uniform_hypergraph_Gnm(3, 5, 4)
        >>> hg.e
        ([(0, 1, 2), (0, 1, 3), (0, 3, 4), (2, 3, 4)], [1.0, 1.0, 1.0, 1.0])
    """
    # similar to UniformRandomUniform in sagemath, https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hypergraph_generators.html

    assert k > 1, "k must be greater than 1"  # TODO ?
    assert num_v > 1, "num_v must be greater than 1"
    assert num_e > 0, "num_e must be greater than 0"

    if n_workers is not None:
        #  use the parallel version for large graph
        edges = set()
        from functools import partial
        from multiprocessing import Pool

        # res_edges = set()
        edges_parallel = split_num_e(num_e=num_e, worker=n_workers)
        local_function = partial(uniform_hypergraph_Gnm_parallel, num_v=num_v, k=k)

        res_edges = set()
        import time

        with Pool(n_workers) as p:
            ret = p.imap(local_function, edges_parallel)
            for res in ret:
                for r in res:
                    res_edges.add(r)

            while len(res_edges) < num_e:
                e = random.sample(range(num_v), k)
                e = tuple(sorted(e))
                if e not in res_edges:
                    res_edges.add(e)

        res_hypergraph = eg.Hypergraph(num_v=num_v, e_list=list(res_edges))
        return res_hypergraph

    else:
        edges = set()
        while len(edges) < num_e:
            e = random.sample(range(num_v), k)
            e = tuple(sorted(e))
            if e not in edges:
                edges.add(e)

    return eg.Hypergraph(num_v, list(edges))


def uniform_hypergraph_configuration_model(k, m, seed=None):
    """
    A function to generate an m-uniform configuration model

    Parameters
    ----------
    k : dictionary
        This is a dictionary where the keys are node ids
        and the values are node degrees.
    m : int
        specifies the hyperedge size
    seed : integer or None (default)
        The seed for the random number generator

    Returns
    -------
    Hypergraph object
        The generated hypergraph

    Warns
    -----
    warnings.warn
        If the sums of the degrees are not divisible by m, the
        algorithm still runs, but raises a warning and adds an
        additional connection to random nodes to satisfy this
        condition.

    Notes
    -----
    This algorithm normally creates multi-edges and loopy hyperedges.
    We remove the loopy hyperedges.

    References
    ----------
    "The effect of heterogeneity on hypergraph contagion models"
    by Nicholas W. Landry and Juan G. Restrepo
    https://doi.org/10.1063/5.0020034


    Example
    -------
    >>> import easygraph as eg
    >>> import random
    >>> n = 1000
    >>> m = 3
    >>> k = {1: 1, 2: 2, 3: 3, 4: 3}
    >>> H = eg.uniform_hypergraph_configuration_model(k, m)

    """
    if seed is not None:
        random.seed(seed)

    # Making sure we have the right number of stubs
    remainder = sum(k.values()) % m
    if remainder != 0:
        warnings.warn(
            "This degree sequence is not realizable. "
            "Increasing the degree of random nodes so that it is."
        )
        random_ids = random.sample(list(k.keys()), int(round(m - remainder)))
        for id in random_ids:
            k[id] = k[id] + 1

    stubs = []
    # Creating the list to index through
    for id in k:
        stubs.extend([id] * int(k[id]))

    H = eg.Hypergraph(num_v=len(k))

    while len(stubs) != 0:
        u = random.sample(range(len(stubs)), m)
        edge = set()
        for index in u:
            edge.add(stubs[index])
        if len(edge) == m:
            H.add_hyperedges(list(edge))

        for index in sorted(u, reverse=True):
            del stubs[index]

    return H


def uniform_HSBM(n, m, p, sizes, seed=None):
    """Create a uniform hypergraph stochastic block model (HSBM).

    Parameters
    ----------
    n : int
        The number of nodes
    m : int
        The hyperedge size
    p : m-dimensional numpy array
        tensor of probabilities between communities
    sizes : list or 1D numpy array
        The sizes of the community blocks in order
    seed : integer or None (default)
        The seed for the random number generator

    Returns
    -------
    Hypergraph
        The constructed SBM hypergraph

    Raises
    ------
    EasyGraphError
        - If the length of sizes and p do not match.
        - If p is not a tensor with every dimension equal
        - If p is not m-dimensional
        - If the entries of p are not in the range [0, 1]
        - If the sum of the vector of sizes does not equal the number of nodes.
    Exception
        If there is an integer overflow error

    See Also
    --------
    uniform_HPPM

    References
    ----------
    Nicholas W. Landry and Juan G. Restrepo.
    "Polarization in hypergraphs with community structure."
    Preprint, 2023. https://doi.org/10.48550/arXiv.2302.13967
    """
    # Check if dimensions match
    if len(sizes) != np.size(p, axis=0):
        raise EasyGraphError("'sizes' and 'p' do not match.")
    if len(np.shape(p)) != m:
        raise EasyGraphError("The dimension of p does not match m")
    # Check that p has the same length over every dimension.
    if len(set(np.shape(p))) != 1:
        raise EasyGraphError("'p' must be a square tensor.")
    if np.max(p) > 1 or np.min(p) < 0:
        raise EasyGraphError("Entries of 'p' not in [0,1].")
    if np.sum(sizes) != n:
        raise EasyGraphError("Sum of sizes does not match n")

    if seed is not None:
        np.random.seed(seed)

    node_labels = range(n)
    H = eg.Hypergraph(num_v=n)

    block_range = range(len(sizes))
    # Split node labels in a partition (list of sets).
    size_cumsum = [sum(sizes[0:x]) for x in range(0, len(sizes) + 1)]
    partition = [
        list(node_labels[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(0, len(size_cumsum) - 1)
    ]

    for block in itertools.product(block_range, repeat=m):
        if p[block] == 1:  # Test edges cases p_ij = 0 or 1
            edges = itertools.product((partition[i] for i in block_range))
            for e in edges:
                H.add_hyperedges(list(e))
        elif p[block] > 0:
            partition_sizes = [len(partition[i]) for i in block]
            max_index = reduce(operator.mul, partition_sizes, 1)
            if max_index < 0:
                raise Exception("Index overflow error!")
            index = np.random.geometric(p[block]) - 1

            while index < max_index:
                indices = _index_to_edge_partition(index, partition_sizes, m)
                e = {partition[block[i]][indices[i]] for i in range(m)}
                if len(e) == m:
                    H.add_hyperedges(list(e))
                index += np.random.geometric(p[block])
    return H


def uniform_HPPM(n, m, rho, k, epsilon, seed=None):
    """Construct the m-uniform hypergraph planted partition model (m-HPPM)

    Parameters
    ----------
    n : int > 0
        Number of nodes
    m : int > 0
        Hyperedge size
    rho : float between 0 and 1
        The fraction of nodes in community 1
    k : float > 0
        Mean degree
    epsilon : float > 0
        Imbalance parameter
    seed : integer or None (default)
        The seed for the random number generator

    Returns
    -------
    Hypergraph
        The constructed m-HPPM hypergraph.

    Raises
    ------
    EasyGraphError
        - If rho is not between 0 and 1
        - If the mean degree is negative.
        - If epsilon is not between 0 and 1

    See Also
    --------
    uniform_HSBM

    References
    ----------
    Nicholas W. Landry and Juan G. Restrepo.
    "Polarization in hypergraphs with community structure."
    Preprint, 2023. https://doi.org/10.48550/arXiv.2302.13967
    """

    if rho < 0 or rho > 1:
        raise EasyGraphError("The value of rho must be between 0 and 1")
    if k < 0:
        raise EasyGraphError("The mean degree must be non-negative")
    if epsilon < 0 or epsilon > 1:
        raise EasyGraphError("epsilon must be between 0 and 1")

    sizes = [int(rho * n), n - int(rho * n)]

    p = k / (m * n ** (m - 1))
    # ratio of inter- to intra-community edges
    q = rho**m + (1 - rho) ** m
    r = 1 / q - 1
    p_in = (1 + r * epsilon) * p
    p_out = (1 - epsilon) * p

    p = p_out * np.ones([2] * m)
    p[tuple([0] * m)] = p_in
    p[tuple([1] * m)] = p_in

    return uniform_HSBM(n, m, p, sizes, seed=seed)


def uniform_erdos_renyi_hypergraph(n, m, p, p_type="degree", seed=None):
    """Generate an m-uniform Erdős–Rényi hypergraph

    This creates a hypergraph with `n` nodes where
    hyperedges of size `m` are created at random to
    obtain a mean degree of `k`.

    Parameters
    ----------
    n : int > 0
        Number of nodes
    m : int > 0
        Hyperedge size
    p : float or int > 0
        Mean expected degree if p_type="degree" and
        probability of an m-hyperedge if p_type="prob"
    p_type : str
        "degree" or "prob", by default "degree"
    seed : integer or None (default)
        The seed for the random number generator

    Returns
    -------
    Hypergraph
        The Erdos Renyi hypergraph


    See Also
    --------
    random_hypergraph
    """
    if seed is not None:
        np.random.seed(seed)

    H = eg.Hypergraph(num_v=n)

    if p_type == "degree":
        q = p / (m * n ** (m - 1))  # wiring probability
    elif p_type == "prob":
        q = p
    else:
        raise EasyGraphError("Invalid p_type!")

    if q > 1 or q < 0:
        raise EasyGraphError("Probability not in [0,1].")

    index = np.random.geometric(q) - 1  # -1 b/c zero indexing
    max_index = n**m
    while index < max_index:
        e = set(_index_to_edge(index, n, m))
        if len(e) == m:
            H.add_hyperedges(list(e))
        index += np.random.geometric(q)
    return H


def _index_to_edge(index, n, m):
    """Generate a hyperedge given an index in the list of possible edges.

    Parameters
    ----------
    index : int > 0
        The index of the hyperedge in the list of all possible hyperedges.
    n : int > 0
        The number of nodes
    m : int > 0
        The hyperedge size.

    Returns
    -------
    list
        The reconstructed hyperedge

    See Also
    --------
    _index_to_edge_partition

    References
    ----------
    https://stackoverflow.com/questions/53834707/element-at-index-in-itertools-product
    """
    return [(index // (n**r) % n) for r in range(m - 1, -1, -1)]


def _index_to_edge_partition(index, partition_sizes, m):
    """Generate a hyperedge given an index in the list of possible edges
    and a partition of community labels.

    Parameters
    ----------
    index : int > 0
        The index of the hyperedge in the list of all possible hyperedges.
    n : int > 0
        The number of nodes
    m : int > 0
        The hyperedge size.

    Returns
    -------
    list
        The reconstructed hyperedge

    See Also
    --------
    _index_to_edge

    """
    try:
        return [
            int(index // np.prod(partition_sizes[r + 1 :]) % partition_sizes[r])
            for r in range(m)
        ]
    except KeyError:
        raise Exception("Invalid parameters")
