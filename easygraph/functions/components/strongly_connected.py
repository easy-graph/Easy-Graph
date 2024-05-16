import easygraph as eg

from easygraph.utils.decorators import *


__all__ = [
    "number_strongly_connected_components",
    "strongly_connected_components",
    "is_strongly_connected",
    "condensation",
]


@not_implemented_for("undirected")
@hybrid("cpp_strongly_connected_components")
def strongly_connected_components(G):
    """Generate nodes in strongly connected components of graph.

    Parameters
    ----------
    G : EasyGraph Graph
        A directed graph.

    Returns
    -------
    comp : generator of sets
        A generator of sets of nodes, one for each strongly connected
        component of G.

    Raises
    ------
    EasyGraphNotImplemented
        If G is undirected.

    Examples
    --------
    Generate a sorted list of strongly connected components, largest first.

    If you only want the largest component, it's more efficient to
    use max instead of sort.

    >>> largest = max(eg.strongly_connected_components(G), key=len)

    See Also
    --------
    connected_components

    Notes
    -----
    Uses Tarjan's algorithm[1]_ with Nuutila's modifications[2]_.
    Nonrecursive version of algorithm.

    References
    ----------
    .. [1] Depth-first search and linear graph algorithms, R. Tarjan
       SIAM Journal of Computing 1(2):146-160, (1972).

    .. [2] On finding the strongly connected components in a directed graph.
       E. Nuutila and E. Soisalon-Soinen
       Information Processing Letters 49(1): 9-14, (1994)..

    """
    preorder = {}
    lowlink = {}
    scc_found = set()
    scc_queue = []
    i = 0  # Preorder counter
    neighbors = {v: iter(G[v]) for v in G}
    for source in G:
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                done = True
                for w in neighbors[v]:
                    if w not in preorder:
                        queue.append(w)
                        done = False
                        break
                if done:
                    lowlink[v] = preorder[v]
                    for w in G[v]:
                        if w not in scc_found:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
                            scc.add(k)
                        scc_found.update(scc)
                        yield scc
                    else:
                        scc_queue.append(v)


def number_strongly_connected_components(G):
    """Returns number of strongly connected components in graph.

    Parameters
    ----------
    G : Easygraph graph
       A directed graph.

    Returns
    -------
    n : integer
       Number of strongly connected components

    Raises
    ------
    EasygraphNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = eg.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (3, 4), (5, 6), (6, 3), (6, 7)])
    >>> eg.number_strongly_connected_components(G)
    3

    See Also
    --------
    strongly_connected_components
    number_connected_components

    Notes
    -----
    For directed graphs only.
    """
    return sum(1 for scc in strongly_connected_components(G))


@not_implemented_for("undirected")
def is_strongly_connected(G):
    """Test directed graph for strong connectivity.

    A directed graph is strongly connected if and only if every vertex in
    the graph is reachable from every other vertex.

    Parameters
    ----------
    G : EasyGraph Graph
       A directed graph.

    Returns
    -------
    connected : bool
      True if the graph is strongly connected, False otherwise.

    Examples
    --------
    >>> G = eg.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 2)])
    >>> eg.is_strongly_connected(G)
    True
    >>> G.remove_edge(2, 3)
    >>> eg.is_strongly_connected(G)
    False

    Raises
    ------
    EasyGraphNotImplemented
        If G is undirected.

    See Also
    --------
    is_connected
    is_biconnected
    strongly_connected_components

    Notes
    -----
    For directed graphs only.
    """
    if len(G) == 0:
        raise eg.EasyGraphPointlessConcept(
            """Connectivity is undefined for the null graph."""
        )

    return len(next(strongly_connected_components(G))) == len(G)


@not_implemented_for("multigraph")
@only_implemented_for_Directed_graph
def condensation(G, scc=None):
    """Returns the condensation of G.
    The condensation of G is the graph with each of the strongly connected
    components contracted into a single node.
    Parameters
    ----------
    G : easygraph.DiGraph
       A directed graph.
    scc:  list or generator (optional, default=None)
       Strongly connected components. If provided, the elements in
       `scc` must partition the nodes in `G`. If not provided, it will be
       calculated as scc=strongly_connected_components(G).
    Returns
    -------
    C : easygraph.DiGraph
       The condensation graph C of G.  The node labels are integers
       corresponding to the index of the component in the list of
       strongly connected components of G.  C has a graph attribute named
       'mapping' with a dictionary mapping the original nodes to the
       nodes in C to which they belong.  Each node in C also has a node
       attribute 'members' with the set of original nodes in G that
       form the SCC that the node in C represents.
    Examples
    --------
    # >>> condensation(G)
    Notes
    -----
    After contracting all strongly connected components to a single node,
    the resulting graph is a directed acyclic graph.
    """
    if scc is None:
        scc = strongly_connected_components(G)
    mapping = {}
    incoming_info = {}
    members = {}
    C = eg.DiGraph()
    # Add mapping dict as graph attribute
    C.graph["mapping"] = mapping
    if len(G) == 0:
        return C
    for i, component in enumerate(scc):
        members[i] = component
        mapping.update((n, i) for n in component)
    number_of_components = i + 1
    for i in range(number_of_components):
        C.add_node(i, member=members[i], incoming=set())
    C.add_nodes(range(number_of_components))
    for edge in G.edges:
        if mapping[edge[0]] != mapping[edge[1]]:
            C.add_edge(mapping[edge[0]], mapping[edge[1]])
            if edge[1] not in incoming_info.keys():
                incoming_info[edge[1]] = set()
            incoming_info[edge[1]].add(edge[0])
    C.graph["incoming_info"] = incoming_info
    return C
