__all__ = ["ego_graph"]

# import easygraph as eg
from easygraph.functions.path import single_source_dijkstra


def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    """Returns induced subgraph of neighbors centered at node n within
    a given radius.

    Parameters
    ----------
    G : graph
      A EasyGraph Graph or DiGraph

    n : node
      A single node

    radius : number, optional
      Include all neighbors of distance<=radius from n.

    center : bool, optional
      If False, do not include center node in graph

    undirected : bool, optional
      If True use both in- and out-neighbors of directed graphs.

    distance : key, optional
      Use specified edge data key as distance.  For example, setting
      distance='weight' will use the edge weight to measure the
      distance from the node n.

    Notes
    -----
    For directed graphs D this produces the "out" neighborhood
    or successors.  If you want the neighborhood of predecessors
    first reverse the graph with D.reverse().  If you want both
    directions use the keyword argument undirected=True.

    Node, edge, and graph attributes are copied to the returned subgraph.
    """
    if undirected:
        """
        if distance is not None:
            sp, _ = eg.single_source_dijkstra(
                G.to_undirected(), n, cutoff=radius, weight=distance
            )
        else:
            sp = dict(
                eg.single_source_shortest_path_length(
                    G.to_undirected(), n, cutoff=radius
                )
            )
        """
    else:
        if distance is not None:
            sp = single_source_dijkstra(G, n, weight=distance)
        else:
            sp = single_source_dijkstra(G, n)
    nodes = [key for key, value in sp.items() if value <= radius]
    nodes = list(nodes)

    H = G.nodes_subgraph(nodes)
    if not center:
        H.remove_node(n)
    return H
