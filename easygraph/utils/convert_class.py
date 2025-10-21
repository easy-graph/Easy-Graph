__all__ = [
    "convert_graph_class",
]


def convert_graph_class(G, graph_class):
    _G = graph_class()
    _G.graph.update(G.graph)
    for node, node_attrs in G.nodes.items():
        dict_attrs = {}
        for key, value in node_attrs:
            dict_attrs[key] = value
        _G.add_node(node, **dict_attrs)
    for u, v, edge_attrs in G.edges:
        dict_attrs = {}
        for key, value in edge_attrs.items():
            dict_attrs[key] = value
        _G.add_edge(u, v, **dict_attrs)
    return _G
