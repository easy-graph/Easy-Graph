__all__ = ["get_relation_of_index_and_node"]


def get_relation_of_index_and_node(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes:
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx
