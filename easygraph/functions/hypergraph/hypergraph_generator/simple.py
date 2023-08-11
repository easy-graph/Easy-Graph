from itertools import combinations

import easygraph as eg

from easygraph.utils.exception import EasyGraphError


__all__ = [
    "star_clique",
]


def star_clique(n_star, n_clique, d_max):
    """Generate a star-clique structure

    That is a star network and a clique network,
    connected by one pairwise edge connecting the centre of the star to the clique.
    network, the each clique is promoted to a hyperedge
    up to order d_max.

    Parameters
    ----------
    n_star : int
        Number of legs of the star
    n_clique : int
        Number of nodes in the clique
    d_max : int
        Maximum order up to which to promote
        cliques to hyperedges

    Returns
    -------
    H : Hypergraph

    Examples
    --------
    >>> import easygraph as eg
    >>> H = eg.star_clique(6, 7, 2)

    Notes
    -----
    The total number of nodes is n_star + n_clique.

    """

    if n_star <= 0:
        raise ValueError("n_star must be an integer > 0.")
    if n_clique <= 0:
        raise ValueError("n_clique must be an integer > 0.")
    if d_max < 0:
        raise ValueError("d_max must be an integer >= 0.")
    elif d_max > n_clique - 1:
        raise ValueError("d_max must be <= n_clique - 1.")

    nodes_star = range(n_star)
    nodes_clique = range(n_star, n_star + n_clique)
    nodes = list(nodes_star) + list(nodes_clique)

    H = eg.Hypergraph(num_v=len(nodes))

    # add star edges (center of the star is 0-th node)
    H.add_hyperedges([[nodes_star[0], nodes_star[i]] for i in range(1, n_star)])

    # connect clique and star by adding last star leg
    H.add_hyperedges([nodes_star[0], nodes_clique[0]])

    # add clique hyperedges up to order d_max

    H.add_hyperedges(
        [
            e
            for d in range(1, d_max + 1)
            for e in list(combinations(nodes_clique, d + 1))
        ]
    )

    return H
