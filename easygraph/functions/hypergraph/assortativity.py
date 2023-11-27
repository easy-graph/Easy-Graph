"""Algorithms for finding the degree assortativity of a hypergraph."""

import random

from itertools import combinations

import numpy
import numpy as np

from easygraph.utils.exception import EasyGraphError


__all__ = ["dynamical_assortativity", "degree_assortativity"]


def dynamical_assortativity(H):
    """Computes the dynamical assortativity of a uniform hypergraph.

    Parameters
    ----------
    H : eg.Hypergraph
        Hypergraph of interest

    Returns
    -------
    float
        The dynamical assortativity

    See Also
    --------
    degree_assortativity

    Raises
    ------
    EasyGraphError
        If the hypergraph is not uniform, or if there are no nodes
        or no edges

    References
    ----------
    Nicholas Landry and Juan G. Restrepo,
    Hypergraph assortativity: A dynamical systems perspective,
    Chaos 2022.
    DOI: 10.1063/5.0086905

    """
    if len(H.v) == 0:
        raise EasyGraphError("Hypergraph must contain nodes")
    elif len(H.e[0]) == 0:
        raise EasyGraphError("Hypergraph must contain edges!")

    if not H.is_uniform():
        raise EasyGraphError("Hypergraph must be uniform!")

    if 1 in H.unique_edge_sizes():
        raise EasyGraphError("No singleton edges!")

    degs = H.deg_v
    k1 = sum(degs) / len(degs)
    k2 = np.mean(numpy.array(degs) ** 2)
    kk1 = np.mean(
        [degs[n1] * degs[n2] for e in H.e[0] for n1, n2 in combinations(e, 2)]
    )

    return kk1 * k1**2 / k2**2 - 1


def degree_assortativity(H, kind="uniform", exact=False, num_samples=1000):
    """Computes the degree assortativity of a hypergraph

    Parameters
    ----------
    H : Hypergraph
        The hypergraph of interest
    kind : str, optional
        the type of degree assortativity. valid choices are
        "uniform", "top-2", and "top-bottom". By default, "uniform".
    exact : bool, optional
        whether to compute over all edges or sample randomly from the
        set of edges. By default, False.
    num_samples : int, optional
        if not exact, specify the number of samples for the computation.
        By default, 1000.

    Returns
    -------
    float
        the degree assortativity

    Raises
    ------
    EasyGraphError
        If there are no nodes or no edges

    See Also
    --------
    dynamical_assortativity

    References
    ----------
    Phil Chodrow,
    Configuration models of random hypergraphs,
    Journal of Complex Networks 2020.
    DOI: 10.1093/comnet/cnaa018
    """

    if len(H.v) == 0:
        raise EasyGraphError("Hypergraph must contain nodes")
    elif len(H.e[0]) == 0:
        raise EasyGraphError("Hypergraph must contain edges!")

    degs = H.deg_v
    if exact:
        k1k2 = [_choose_degrees(e, degs, kind) for e in H.e[0] if len(e) > 1]
    else:
        edges = [e for e in H.e[0] if len(e) > 1]
        k1k2 = [
            _choose_degrees(random.choice(H.e[0]), degs, kind)
            for _ in range(num_samples)
        ]

    rho = np.corrcoef(np.array(k1k2).T)[0, 1]
    if np.isnan(rho):
        return 0
    return rho


def _choose_degrees(e, k, kind="uniform"):
    """Choose the degrees of two nodes in a hyperedge.

    Parameters
    ----------
    e : iterable
        the members in a hyperedge
    k : dict
        the degrees where keys are node IDs and values are degrees
    kind : str, optional
        the type of degree assortativity, options are "uniform", "top-2",
        and "top-bottom". By default, "uniform".

    Returns
    -------
    tuple
        two degrees selected from the edge

    Raises
    ------
    EasyGraphError
        if invalid assortativity function chosen

    See Also
    --------
    degree_assortativity

    References
    ----------
    Phil Chodrow,
    Configuration models of random hypergraphs,
    Journal of Complex Networks 2020.
    DOI: 10.1093/comnet/cnaa018
    """
    e = list(e)
    if len(e) > 1:
        if kind == "uniform":
            i = np.random.randint(len(e))
            j = i
            while i == j:
                j = np.random.randint(len(e))
            return (k[e[i]], k[e[j]])

        elif kind == "top-2":
            degs = sorted([k[i] for i in e])[-2:]
            random.shuffle(degs)
            return degs

        elif kind == "top-bottom":
            # this selects the largest and smallest degrees in one line
            degs = sorted([k[i] for i in e])[:: len(e) - 1]
            random.shuffle(degs)
            return degs

        else:
            raise EasyGraphError("Invalid choice function!")
    else:
        raise EasyGraphError("Edge must have more than one member!")
