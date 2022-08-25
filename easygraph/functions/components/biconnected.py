from itertools import chain

from easygraph.utils import *


__all__ = [
    "is_biconnected",
    "biconnected_components",
    "generator_biconnected_components_nodes",
    "generator_biconnected_components_edges",
    "generator_articulation_points",
]


@not_implemented_for("multigraph", "directed")
def is_biconnected(G):
    """Returns whether the graph is biconnected or not.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    Returns
    -------
    is_biconnected : boolean
        `True` if the graph is biconnected.

    Examples
    --------

    >>> is_biconnected(G)

    """
    bc_nodes = list(generator_biconnected_components_nodes(G))
    if len(bc_nodes) == 1:
        return len(bc_nodes[0]) == len(
            G
        )  # avoid situations where there is isolated vertex
    return False


@not_implemented_for("multigraph", "directed")
# TODO: get the subgraph of each biconnected graph
def biconnected_components(G):
    """Returns a list of biconnected components, each of which denotes the edges set of a biconnected component.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    Returns
    -------
    biconnected_components : list of list
        Each element list is the edges set of a biconnected component.

    Examples
    --------
    >>> connected_components(G)

    """
    return list(generator_biconnected_components_edges(G))


@not_implemented_for("multigraph", "directed")
def generator_biconnected_components_nodes(G):
    """Returns a generator of nodes in each biconnected component.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    Returns
    -------
    Yields nodes set of each biconnected component.

    See Also
    --------
    generator_biconnected_components_edges

    Examples
    --------
    >>> generator_biconnected_components_nodes(G)


    """
    for component in _biconnected_dfs_record_edges(G, need_components=True):
        # TODO: only one edge = biconnected_component?
        yield set(chain.from_iterable(component))


@not_implemented_for("multigraph", "directed")
def generator_biconnected_components_edges(G):
    """Returns a generator of nodes in each biconnected component.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    Returns
    -------
    Yields edges set of each biconnected component.

    See Also
    --------
    generator_biconnected_components_nodes

    Examples
    --------
    >>> generator_biconnected_components_edges(G)

    """
    yield from _biconnected_dfs_record_edges(G, need_components=True)


@not_implemented_for("multigraph", "directed")
def generator_articulation_points(G):
    """Returns a generator of articulation points.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    Returns
    -------
    Yields the articulation point in *G*.

    Examples
    --------
    >>> generator_articulation_points(G)

    """
    seen = set()
    for cut_vertex in _biconnected_dfs_record_edges(G, need_components=False):
        if cut_vertex not in seen:
            seen.add(cut_vertex)
            yield cut_vertex


@hybrid("cpp_biconnected_dfs_record_edges")
def _biconnected_dfs_record_edges(G, need_components=True):
    """
    References
    ----------
    https://www.cnblogs.com/nullzx/p/7968110.html
    https://blog.csdn.net/gauss_acm/article/details/43493903
    """
    # record edges of each biconnected component in traversal
    # Copied version from EasyGraph
    # depth-first search algorithm to generate articulation points
    # and biconnected components
    visited = set()
    for start in G:
        if start in visited:
            continue
        discovery = {start: 0}  # time of first discovery of node during search
        low = {start: 0}
        root_children = 0
        visited.add(start)
        edge_stack = []
        stack = [(start, start, iter(G[start]))]
        while stack:
            grandparent, parent, children = stack[-1]
            try:
                child = next(children)
                if grandparent == child:
                    continue
                if child in visited:
                    if discovery[child] <= discovery[parent]:  # back edge
                        low[parent] = min(low[parent], discovery[child])
                        if need_components:
                            edge_stack.append((parent, child))
                else:
                    low[child] = discovery[child] = len(discovery)
                    visited.add(child)
                    stack.append((parent, child, iter(G[child])))
                    if need_components:
                        edge_stack.append((parent, child))
            except StopIteration:
                stack.pop()
                if len(stack) > 1:
                    if low[parent] >= discovery[grandparent]:
                        if need_components:
                            ind = edge_stack.index((grandparent, parent))
                            yield edge_stack[ind:]
                            edge_stack = edge_stack[:ind]
                        else:
                            yield grandparent
                    low[grandparent] = min(low[parent], low[grandparent])
                elif stack:  # length 1 so grandparent is root
                    root_children += 1
                    if need_components:
                        ind = edge_stack.index((grandparent, parent))
                        yield edge_stack[ind:]
        if not need_components:
            # root node is articulation point if it has more than 1 child
            if root_children > 1:
                yield start


def _biconnected_dfs_record_nodes(G, need_components=True):
    # record nodes of each biconnected component in traversal
    # Not used.
    visited = set()
    for start in G:
        if start in visited:
            continue
        discovery = {start: 0}  # time of first discovery of node during search
        low = {start: 0}
        root_children = 0
        visited.add(start)
        node_stack = [start]
        stack = [(start, start, iter(G[start]))]
        while stack:
            grandparent, parent, children = stack[-1]
            try:
                child = next(children)
                if grandparent == child:
                    continue
                if child in visited:
                    if discovery[child] <= discovery[parent]:  # back edge
                        low[parent] = min(low[parent], discovery[child])
                else:
                    low[child] = discovery[child] = len(discovery)
                    visited.add(child)
                    stack.append((parent, child, iter(G[child])))
                    if need_components:
                        node_stack.append(child)
            except StopIteration:
                stack.pop()
                if len(stack) > 1:
                    if low[parent] >= discovery[grandparent]:
                        if need_components:
                            ind = node_stack.index(grandparent)
                            yield node_stack[ind:]
                            node_stack = node_stack[: ind + 1]
                        else:
                            yield grandparent
                    low[grandparent] = min(low[parent], low[grandparent])
                elif stack:  # length 1 so grandparent is root
                    root_children += 1
                    if need_components:
                        ind = node_stack.index(grandparent)
                        yield node_stack[ind:]
        if not need_components:
            # root node is articulation point if it has more than 1 child
            if root_children > 1:
                yield start
