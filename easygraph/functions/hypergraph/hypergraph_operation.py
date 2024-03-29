from easygraph.exception import EasyGraphError


__all__ = [
    "hypergraph_density",
]


def hypergraph_density(hg, ignore_singletons=False):
    r"""Hypergraph density.

    The density of a hypergraph is the number of existing edges divided by the number of
    possible edges.

    Let `H` have :math:`n` nodes and :math:`m` hyperedges. Then,

    * `density(H) =` :math:`\frac{m}{2^n - 1}`,
    * `density(H, ignore_singletons=True) =` :math:`\frac{m}{2^n - 1 - n}`.

    Here, :math:`2^n` is the total possible number of hyperedges on `H`, from which we
    subtract :math:`1` because the empty hyperedge is not considered.  We subtract an
    additional :math:`n` when singletons are not considered.

    Now assume `H` has :math:`a` edges with order :math:`1` and :math:`b` edges with
    order :math:`2`.  Then,

    * `density(H, order=1) =` :math:`\frac{a}{{n \choose 2}}`,
    * `density(H, order=2) =` :math:`\frac{b}{{n \choose 3}}`,
    * `density(H, max_order=1) =` :math:`\frac{a}{{n \choose 1} + {n \choose 2}}`,
    * `density(H, max_order=1, ignore_singletons=True) =` :math:`\frac{a}{{n \choose 2}}`,
    * `density(H, max_order=2) =` :math:`\frac{m}{{n \choose 1} + {n \choose 2} + {n \choose 3}}`,
    * `density(H, max_order=2, ignore_singletons=True) =` :math:`\frac{m}{{n \choose 2} + {n \choose 3}}`,

    Parameters
    ---------
    order : int, optional
        If not None, only count edges of the specified order.
        By default, None.

    max_order : int, optional
        If not None, only count edges of order up to this value, inclusive.
        By default, None.

    ignore_singletons : bool, optional
        Whether to consider singleton edges.  Ignored if `order` is not None and
        different from :math:`0`. By default, False.

    See Also
    --------
    :func:`incidence_density`

    Notes
    -----
    If both `order` and `max_order` are not None, `max_order` is ignored.

    """
    n = hg.num_v
    numer = len(hg.e[0])
    if n < 1:
        raise EasyGraphError("Density not defined for empty hypergraph")
    if numer < 1:
        return 0.0

    denom = 2**n - 1
    if ignore_singletons:
        denom -= n
    try:
        return numer / float(denom)
    except ZeroDivisionError:
        return 0.0
