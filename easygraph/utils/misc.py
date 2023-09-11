from collections.abc import Iterable
from collections.abc import Iterator
from itertools import chain
from itertools import tee


__all__ = [
    "split_len",
    "split",
    "nodes_equal",
    "edges_equal",
    "pairwise",
    "graphs_equal",
    # "arbitrary_element"
]


def split_len(nodes, step=30000):
    ret = []
    length = len(nodes)
    for i in range(0, length, step):
        ret.append(nodes[i : i + step])
    if len(ret[-1]) * 3 < step:
        ret[-2] = ret[-2] + ret[-1]
        ret = ret[:-1]
    return ret


def split(nodes, n):
    ret = []
    length = len(nodes)  # 总长
    step = int(length / n) + 1  # 每份的长度
    for i in range(0, length, step):
        ret.append(nodes[i : i + step])
    return ret


def nodes_equal(nodes1, nodes2):
    """Check if nodes are equal.

    Equality here means equal as Python objects.
    Node data must match if included.
    The order of nodes is not relevant.

    Parameters
    ----------
    nodes1, nodes2 : iterables of nodes, or (node, datadict) tuples

    Returns
    -------
    bool
        True if nodes are equal, False otherwise.
    """
    nlist1 = list(nodes1)
    nlist2 = list(nodes2)
    try:
        d1 = dict(nlist1)
        d2 = dict(nlist2)
    except (ValueError, TypeError):
        d1 = dict.fromkeys(nlist1)
        d2 = dict.fromkeys(nlist2)
    return d1 == d2


def edges_equal(edges1, edges2, need_data=True):
    """Check if edges are equal.

    Equality here means equal as Python objects.
    Edge data must match if included.
    The order of the edges is not relevant.

    Parameters
    ----------
    edges1, edges2 : iterables of with u, v nodes as
        edge tuples (u, v), or
        edge tuples with data dicts (u, v, d), or
        edge tuples with keys and data dicts (u, v, k, d)

    Returns
    -------
    bool
        True if edges are equal, False otherwise.
    """
    from collections import defaultdict

    d1 = defaultdict(dict)
    d2 = defaultdict(dict)
    c1 = 0
    for c1, e in enumerate(edges1):
        u, v = e[0], e[1]
        data = []
        if need_data == True:
            data = [e[2:]]
            if v in d1[u]:
                data = d1[u][v] + data
        d1[u][v] = data
        d1[v][u] = data
    c2 = 0
    for c2, e in enumerate(edges2):
        u, v = e[0], e[1]
        data = []
        if need_data == True:
            data = [e[2:]]
            if v in d2[u]:
                data = d2[u][v] + data
        d2[u][v] = data
        d2[v][u] = data
    if c1 != c2:
        return False
    # can check one direction because lengths are the same.
    for n, nbrdict in d1.items():
        for nbr, datalist in nbrdict.items():
            if n not in d2:
                return False
            if nbr not in d2[n]:
                return False
            d2datalist = d2[n][nbr]
            for data in datalist:
                if datalist.count(data) != d2datalist.count(data):
                    return False
    return True


# Recipe from the itertools documentation.
def pairwise(iterable, cyclic=False):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    first = next(b, None)
    if cyclic is True:
        return zip(a, chain(b, (first,)))
    return zip(a, b)


def graphs_equal(graph1, graph2):
    """Check if graphs are equal.

    Equality here means equal as Python objects (not isomorphism).
    Node, edge and graph data must match.

    Parameters
    ----------
    graph1, graph2 : graph

    Returns
    -------
    bool
        True if graphs are equal, False otherwise.
    """
    return (
        graph1.adj == graph2.adj
        and graph1.nodes == graph2.nodes
        and graph1.graph == graph2.graph
    )


# def arbitrary_element(iterable):
#     """Returns an arbitrary element of `iterable` without removing it.

#     This is most useful for "peeking" at an arbitrary element of a set,
#     but can be used for any list, dictionary, etc., as well.

#     Parameters
#     ----------
#     iterable : `abc.collections.Iterable` instance
#         Any object that implements ``__iter__``, e.g. set, dict, list, tuple,
#         etc.

#     Returns
#     -------
#     The object that results from ``next(iter(iterable))``

#     Raises
#     ------
#     ValueError
#         If `iterable` is an iterator (because the current implementation of
#         this function would consume an element from the iterator).

#     Examples
#     --------
#     Arbitrary elements from common Iterable objects:

#     >>> eg.utils.arbitrary_element([1, 2, 3])  # list
#     1
#     >>> eg.utils.arbitrary_element((1, 2, 3))  # tuple
#     1
#     >>> eg.utils.arbitrary_element({1, 2, 3})  # set
#     1
#     >>> d = {k: v for k, v in zip([1, 2, 3], [3, 2, 1])}
#     >>> eg.utils.arbitrary_element(d)  # dict_keys
#     1
#     >>> eg.utils.arbitrary_element(d.values())   # dict values
#     3

#     `str` is also an Iterable:

#     >>> eg.utils.arbitrary_element("hello")
#     'h'

#     :exc:`ValueError` is raised if `iterable` is an iterator:

#     >>> iterator = iter([1, 2, 3])  # Iterator, *not* Iterable
#     >>> eg.utils.arbitrary_element(iterator)
#     Traceback (most recent call last):
#         ...
#     ValueError: cannot return an arbitrary item from an iterator

#     Notes
#     -----
#     This function does not return a *random* element. If `iterable` is
#     ordered, sequential calls will return the same value::

#         >>> l = [1, 2, 3]
#         >>> eg.utils.arbitrary_element(l)
#         1
#         >>> eg.utils.arbitrary_element(l)
#         1

#     """
#     if isinstance(iterable, Iterator):
#         raise ValueError("cannot return an arbitrary item from an iterator")
#     # Another possible implementation is ``for x in iterable: return x``.
#     return next(iter(iterable))
