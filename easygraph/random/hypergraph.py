import random

from easygraph.classes.graph import Graph


__all__ = ["graph_Gnm"]


def graph_Gnm(num_v: int, num_e: int):
    r"""Return a random graph with ``num_v`` verteices and ``num_e`` edges. Edges are drawn uniformly from the set of possible edges.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``num_e`` (``int``): The Number of edges.

    Examples:
        >>> import easygraph.random as random
        >>> g = random.graph_Gnm(4, 5)
        >>> g.e
        ([(1, 2), (0, 3), (2, 3), (0, 2), (1, 3)], [1.0, 1.0, 1.0, 1.0, 1.0])
    """
    assert num_v > 1, "num_v must be greater than 1"
    assert (
        num_e < num_v * (num_v - 1) // 2
    ), "the specified num_e is larger than the possible number of edges"

    v_list = list(range(num_v))
    cur_num_e, e_set = 0, set()
    while cur_num_e < num_e:
        v = random.choice(v_list)
        w = random.choice(v_list)
        if v > w:
            v, w = w, v
        if v == w or (v, w) in e_set:
            continue
        e_set.add((v, w))
        cur_num_e += 1
    g = Graph()
    g.add_nodes(list(range(0, num_v)))
    for ee in list(e_set):
        g.add_edge(ee[0], ee[1], weight=1.0)

    return g
