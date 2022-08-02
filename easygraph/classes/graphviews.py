from easygraph.utils import only_implemented_for_Directed_graph


__all__ = ["reverse_view"]


@only_implemented_for_Directed_graph
def reverse_view(G):
    newG = G.__class__()
    newG._graph = G
    newG.graph = G.graph
    newG._node = G._node
    newG._succ, newG._pred = G._pred, G._succ
    newG._adj = newG._succ
    return newG
