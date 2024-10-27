"""Algorithms for computing nodal clustering coefficients."""

import numpy as np

from easygraph.utils.exception import EasyGraphError


__all__ = [
    "hypergraph_clustering_coefficient",
    "hypergraph_local_clustering_coefficient",
    "hypergraph_two_node_clustering_coefficient",
]


def hypergraph_clustering_coefficient(H):
    r"""Return the clustering coefficients for
    each node in a Hypergraph.

    This clustering coefficient is defined as the
    clustering coefficient of the unweighted pairwise
    projection of the hypergraph, i.e.,
    :math:`c = A^3_{i,i}/\binom{k}{2},`
    where :math:`A` is the adjacency matrix of the network
    and :math:`k` is the pairwise degree of :math:`i`.

    Parameters
    ----------
    H : Hypergraph
        Hypergraph

    Returns
    -------
    dict
        nodes are keys, clustering coefficients are values.

    Notes
    -----
    The clustering coefficient is undefined when the number of
    neighbors is 0 or 1, but we set the clustering coefficient
    to 0 in these cases. For more discussion, see
    https://arxiv.org/abs/0802.2512

    See Also
    --------
    local_clustering_coefficient
    two_node_clustering_coefficient

    References
    ----------
    "Clustering Coefficients in Protein Interaction Hypernetworks"
    by Suzanne Gallagher and Debra Goldberg.
    DOI: 10.1145/2506583.2506635

    Example
    -------
    >>> import easygraph as eg
    >>> H = eg.random_hypergraph(3, [1, 1])
    >>> cc = eg.clustering_coefficient(H)
    >>> cc
    {0: 1.0, 1: 1.0, 2: 1.0}
    """
    adj = H.adjacency_matrix()
    k = np.array(adj.sum(axis=1))
    l = []
    for i in k:
        l.append(i[0])
    k = np.array(l)
    denom = k * (k - 1) / 2
    mat = adj.dot(adj).dot(adj)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.nan_to_num(0.5 * mat.diagonal() / denom)
    r = {}
    for i in range(0, len(H.v)):
        r[i] = result[i]
    return r


def hypergraph_local_clustering_coefficient(H):
    """Compute the local clustering coefficient.

    This clustering coefficient is based on the
    overlap of the edges connected to a given node,
    normalized by the size of the node's neighborhood.

    Parameters
    ----------
    H : Hypergraph
        Hypergraph

    Returns
    -------
    dict
        keys are node IDs and values are the
        clustering coefficients.

    Notes
    -----
    The clustering coefficient is undefined when the number of
    neighbors is 0 or 1, but we set the clustering coefficient
    to 0 in these cases. For more discussion, see
    https://arxiv.org/abs/0802.2512

    See Also
    --------
    clustering_coefficient
    two_node_clustering_coefficient

    References
    ----------
    "Properties of metabolic graphs: biological organization or representation
    artifacts?"  by Wanding Zhou and Luay Nakhleh.
    https://doi.org/10.1186/1471-2105-12-132

    "Hypergraphs for predicting essential genes using multiprotein complex data"
    by Florian Klimm, Charlotte M. Deane, and Gesine Reinert.
    https://doi.org/10.1093/comnet/cnaa028

    Example
    -------
    >>> import easygraph as eg
    >>> H = eg.random_hypergraph(3, [1, 1])
    >>> cc = eg.hypergraph_local_clustering_coefficient(H)
    >>> cc
    {0: 1.0, 1: 1.0, 2: 1.0}

    """
    result = {}
    # 节点属于哪些边
    memberships = []
    for n in H.v:
        tmp = set()
        for index, e in enumerate(H.e[0]):
            if n in e:
                tmp.add(index)
        memberships.append(tmp)

    # 每条边包含哪些节点
    members = H.e[0]
    for n in H.v:
        ev = memberships[n]
        dv = len(ev)
        if dv <= 1:
            result[n] = 0
        else:
            total_eo = 0
            # go over all pairs of edges pairwise
            for e1 in range(dv):
                edge1 = members[e1]
                for e2 in range(e1):
                    edge2 = members[e2]
                    # set differences for the hyperedges
                    D1 = set(edge1) - set(edge2)
                    D2 = set(edge2) - set(edge1)
                    # if edges are the same by definition the extra overlap is zero
                    if len(D1.union(D2)) == 0:
                        eo = 0
                    else:
                        # otherwise we have to look at their neighbors
                        # the neighbors of D1 and D2, respectively.
                        neighD1 = {i for d in D1 for i in H.neighbor_of_node(d)}
                        neighD2 = {i for d in D2 for i in H.neighbor_of_node(d)}
                        # compute extra overlap [len() is used for cardinality of edges]
                        eo = (
                            len(neighD1.intersection(D2))
                            + len(neighD2.intersection(D1))
                        ) / len(
                            D1.union(D2)
                        )  # add it up
                    # add it up
                    total_eo = total_eo + eo

            # include normalization by degree k*(k-1)/2
            result[n] = 2 * total_eo / (dv * (dv - 1))
    return result


def hypergraph_two_node_clustering_coefficient(H, kind="union"):
    """Return the clustering coefficients for
    each node in a Hypergraph.

    This definition averages over all of the
    two-node clustering coefficients involving the node.

    Parameters
    ----------
    H : Hypergraph
        Hypergraph
    kind : string, optional
        The type of two node clustering coefficient. Options
        are "union", "max", and "min". By default, "union".

    Returns
    -------
    dict
        nodes are keys, clustering coefficients are values.

    Notes
    -----
    The clustering coefficient is undefined when the number of
    neighbors is 0 or 1, but we set the clustering coefficient
    to 0 in these cases. For more discussion, see
    https://arxiv.org/abs/0802.2512

    See Also
    --------
    clustering_coefficient
    local_clustering_coefficient

    References
    ----------
    "Clustering Coefficients in Protein Interaction Hypernetworks"
    by Suzanne Gallagher and Debra Goldberg.
    DOI: 10.1145/2506583.2506635

    Example
    -------
    >>> import easygraph as eg
    >>> H = eg.random_hypergraph(3, [1, 1])
    >>> cc = eg.two_node_clustering_coefficient(H, kind="union")
    >>> cc
    {0: 0.5, 1: 0.5, 2: 0.5}
    """
    result = {}
    memberships = {}
    for n in H.v:
        tmp = set()
        for index, e in enumerate(H.e[0]):
            if n in e:
                tmp.add(index)
        memberships[n] = tmp

    for n in H.v:
        neighbors = H.neighbor_of_node(n)
        result[n] = 0.0
        for v in neighbors:
            result[n] += _uv_cc(n, v, memberships, kind=kind) / len(neighbors)
    return result


def _uv_cc(u, v, memberships, kind="union"):
    """Helper function to compute the two-node
    clustering coefficient.

    Parameters
    ----------
    u : hashable
        First node
    v : hashable
        Second node
    memberships : dict
        node IDs are keys, edge IDs to which they belong
        are values.
    kind : str, optional
        Type of clustering coefficient to compute, by default "union".
        Options:

        - "union"
        - "max"
        - "min"

    Returns
    -------
    float
        The clustering coefficient

    Raises
    ------
    EasyGraphError
        If an invalid clustering coefficient kind
        is specified.

    References
    ----------
    "Clustering Coefficients in Protein Interaction Hypernetworks"
    by Suzanne Gallagher and Debra Goldberg.
    DOI: 10.1145/2506583.2506635
    """
    m_u = memberships[u]
    m_v = memberships[v]

    num = len(m_u.intersection(m_v))

    if kind == "union":
        denom = len(m_u.union(m_v))
    elif kind == "min":
        denom = min(len(m_u), len(m_v))
    elif kind == "max":
        denom = max(len(m_u), len(m_v))
    else:
        raise EasyGraphError("Invalid kind of clustering.")

    if denom == 0:
        return np.nan

    return num / denom
