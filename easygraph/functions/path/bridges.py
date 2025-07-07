from itertools import chain

import easygraph as eg

from easygraph.utils.decorators import *


__all__ = ["bridges", "has_bridges"]


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
def bridges(G, root=None):
    """Generate all bridges in a graph.

    A *bridge* in a graph is an edge whose removal causes the number of
    connected components of the graph to increase.  Equivalently, a bridge is an
    edge that does not belong to any cycle.

    Parameters
    ----------
    G : undirected graph

    root : node (optional)
       A node in the graph `G`. If specified, only the bridges in the
       connected component containing this node will be returned.

    Yields
    ------
    e : edge
       An edge in the graph whose removal disconnects the graph (or
       causes the number of connected components to increase).

    Raises
    ------
    NodeNotFound
       If `root` is not in the graph `G`.

    Examples
    --------

    >>> list(eg.bridges(G))
    [(9, 10)]

    Notes
    -----
    This is an implementation of the algorithm described in _[1].  An edge is a
    bridge if and only if it is not contained in any chain. Chains are found
    using the :func:`chain_decomposition` function.

    Ignoring polylogarithmic factors, the worst-case time complexity is the
    same as the :func:`chain_decomposition` function,
    $O(m + n)$, where $n$ is the number of nodes in the graph and $m$ is
    the number of edges.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bridge_%28graph_theory%29#Bridge-Finding_with_Chain_Decompositions
    """
    if root is not None and root not in G.nodes:
        raise eg.NodeNotFound(f"Node {root} is not in the graph.")
    chains = chain_decomposition(G, root=root)
    chain_edges = set(chain.from_iterable(chains))
    for u, v, t in G.edges:
        if (u, v) not in chain_edges and (v, u) not in chain_edges:
            yield u, v


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
def has_bridges(G, root=None):
    """Decide whether a graph has any bridges.

    A *bridge* in a graph is an edge whose removal causes the number of
    connected components of the graph to increase.

    Parameters
    ----------
    G : undirected graph

    root : node (optional)
       A node in the graph `G`. If specified, only the bridges in the
       connected component containing this node will be considered.

    Returns
    -------
    bool
       Whether the graph (or the connected component containing `root`)
       has any bridges.

    Raises
    ------
    NodeNotFound
       If `root` is not in the graph `G`.

    Examples
    --------

    >>> eg.has_bridges(G)
    True

    Notes
    -----
    This implementation uses the :func:`easygraph.bridges` function, so
    it shares its worst-case time complexity, $O(m + n)$, ignoring
    polylogarithmic factors, where $n$ is the number of nodes in the
    graph and $m$ is the number of edges.

    """
    try:
        next(bridges(G, root))
    except StopIteration:
        return False
    else:
        return True


def chain_decomposition(G, root=None):
    def _dfs_cycle_forest(G, root=None):
        H = eg.DiGraph()
        nodes = []
        for u, v, d in dfs_labeled_edges(G, source=root):
            if d == "forward":
                # `dfs_labeled_edges()` yields (root, root, 'forward')
                # if it is beginning the search on a new connected
                # component.
                if u == v:
                    H.add_node(v, parent=None)
                    nodes.append(v)
                else:
                    H.add_node(v, parent=u)
                    H.add_edge(v, u, nontree=False)
                    nodes.append(v)
            # `dfs_labeled_edges` considers nontree edges in both
            # orientations, so we need to not add the edge if it its
            # other orientation has been added.
            elif d == "nontree" and v not in H[u]:
                H.add_edge(v, u, nontree=True)
            else:
                # Do nothing on 'reverse' edges; we only care about
                # forward and nontree edges.
                pass
        return H, nodes

    def _build_chain(G, u, v, visited):
        while v not in visited:
            yield u, v
            visited.add(v)
            u, v = v, G.nodes[v]["parent"]
        yield u, v

    H, nodes = _dfs_cycle_forest(G, root)

    visited = set()
    for u in nodes:
        visited.add(u)
        # For each nontree edge going out of node u...
        edges = []
        for w, v, d in H.edges:
            if w == u and d["nontree"] == True:
                edges.append((w, v))
        # edges = ((u, v) for u, v, d in H.out_edges(u, data="nontree") if d)
        for u, v in edges:
            # Create the cycle or cycle prefix starting with the
            # nontree edge.
            chain = list(_build_chain(H, u, v, visited))
            yield chain


def dfs_labeled_edges(G, source=None, depth_limit=None):
    if source is None:
        # edges for all components
        nodes = G
    else:
        # edges for components with source
        nodes = [source]
    visited = set()
    if depth_limit is None:
        depth_limit = len(G)
    for start in nodes:
        if start in visited:
            continue
        yield start, start, "forward"
        visited.add(start)
        stack = [(start, depth_limit, iter(G[start]))]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child in visited:
                    yield parent, child, "nontree"
                else:
                    yield parent, child, "forward"
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, iter(G[child])))
            except StopIteration:
                stack.pop()
                if stack:
                    yield stack[-1][0], parent, "reverse"
        yield start, start, "reverse"
