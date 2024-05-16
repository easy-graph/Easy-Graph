from easygraph.utils.decorators import *


__all__ = [
    "is_connected",
    "number_connected_components",
    "connected_components",
    "connected_components_directed",
    "connected_component_of_node",
]


@not_implemented_for("multigraph")
def is_connected(G):
    """Returns whether the graph is connected or not.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    Returns
    -------
    is_biconnected : boolean
        `True` if the graph is connected.

    Examples
    --------

    >>> is_connected(G)

    """
    assert len(G) != 0, "No node in the graph."
    arbitrary_node = next(iter(G))  # Pick an arbitrary node to run BFS
    return len(G) == sum(1 for node in _plain_bfs(G, arbitrary_node))


@not_implemented_for("multigraph")
def number_connected_components(G):
    """Returns the number of connected components.

    Parameters
    ----------
    G : easygraph.Graph

    Returns
    -------
    number_connected_components : int
        The number of connected components.

    Examples
    --------
    >>> number_connected_components(G)

    """
    return sum(1 for component in _generator_connected_components(G))


@not_implemented_for("multigraph")
@hybrid("cpp_connected_components_undirected")
def connected_components(G):
    """Returns a list of connected components, each of which denotes the edges set of a connected component.

    Parameters
    ----------
    G : easygraph.Graph
    Returns
    -------
    connected_components : list of list
        Each element list is the edges set of a connected component.

    Examples
    --------
    >>> connected_components(G)

    """
    seen = set()
    for v in G:
        if v not in seen:
            c = set(_plain_bfs(G, v))
            seen.update(c)
            yield c


@not_implemented_for("multigraph")
@hybrid("cpp_connected_components_directed")
def connected_components_directed(G):
    """Returns a list of connected components, each of which denotes the edges set of a connected component.

    Parameters
    ----------
    G :  easygraph.DiGraph
    Returns
    -------
    connected_components : list of list
        Each element list is the edges set of a connected component.

    Examples
    --------
    >>> connected_components(G)

    """
    seen = set()
    for v in G:
        if v not in seen:
            c = set(_plain_bfs(G, v))
            seen.update(c)
            yield c


def _generator_connected_components(G):
    seen = set()
    for v in G:
        if v not in seen:
            component = set(_plain_bfs(G, v))
            yield component
            seen.update(component)


@not_implemented_for("multigraph")
def connected_component_of_node(G, node):
    """Returns the connected component that *node* belongs to.

    Parameters
    ----------
    G : easygraph.Graph

    node : object
        The target node

    Returns
    -------
    connected_component_of_node : set
        The connected component that *node* belongs to.

    Examples
    --------
    Returns the connected component of one node `Jack`.

    >>> connected_component_of_node(G, node='Jack')

    """
    return set(_plain_bfs(G, node))


@hybrid("cpp_plain_bfs")
def _plain_bfs(G, source):
    """
    A fast BFS node generator
    """
    G_adj = G.adj
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(G_adj[v])
