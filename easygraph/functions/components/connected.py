import sys
sys.path.append('../../../')
import easygraph
from easygraph.utils.decorators import only_implemented_for_UnDirected_graph
from threading import Thread

__all__ = [
    "is_connected",
    "number_connected_components",
    "connected_components",
    "connected_component_of_node"
]


@only_implemented_for_UnDirected_graph
def is_connected(G):
    assert len(G) != 0, "No node in the graph."
    arbitrary_node = next(iter(G))  # Pick an arbitrary node to run BFS
    return len(G) == sum(1 for node in _plain_bfs(G, arbitrary_node))

@only_implemented_for_UnDirected_graph
def number_connected_components(G):
    return sum(1 for component in _generator_connected_components(G))


@only_implemented_for_UnDirected_graph
def connected_components(G):
    # Return all components ordered by number of nodes included
    all_components = sorted(list(_generator_connected_components(G)), key=len)
    return all_components


@only_implemented_for_UnDirected_graph
def _generator_connected_components(G):
    seen = set()
    for v in G:
        if v not in seen:
            component = set(_plain_bfs(G, v))
            yield component
            seen.update(component)

@only_implemented_for_UnDirected_graph
def connected_component_of_node(G, node):
    return set(_plain_bfs(G, node))


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
