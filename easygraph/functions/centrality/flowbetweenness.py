import collections
import copy

from easygraph.utils.decorators import *


__all__ = [
    "flowbetweenness_centrality",
]


@not_implemented_for("multigraph")
def flowbetweenness_centrality(G):
    """Compute the independent-basic betweenness centrality for nodes in a flow network.

    .. math::

       c_B(v) =\\sum_{s,t \\in V} \frac{\\sigma(s, t|v)}{\\sigma(s, t)}

    where V is the set of nodes,

    .. math::

        \\sigma(s, t)\\ is\\ the\\ number\\ of\\ independent\\ (s, t)-paths,

    .. math::

        \\sigma(s, t|v)\\ is\\ the\\ maximum\\ number\\ possible\\ of\\ those\\ paths\\ passing\\ through\\ some\\ node\\ v\\ other\\ than\\ s, t.\

    .. math::

        If\\ s\\ =\\ t,\\ \\sigma(s, t)\\ =\\ 1,\\ and\\ if\\ v \\in \\{s, t\\},\\ \\sigma(s, t|v)\\ =\\ 0\\ [2]_.

    Parameters
    ----------
    G : graph
      A easygraph directed graph.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with independent-basic betweenness centrality as the value.

    Notes
    -----
    A flow network is a directed graph where each edge has a capacity and each edge receives a flow.
    """
    if G.is_directed() == False:
        print("Please input a directed graph")
        return
    flow_dict = NumberOfFlow(G)
    nodes = G.nodes
    result_dict = dict()
    for node, _ in nodes.items():
        result_dict[node] = 0
    for node_v, _ in nodes.items():
        for node_s, _ in nodes.items():
            for node_t, _ in nodes.items():
                num = 1
                num_v = 0
                if node_s == node_t:
                    num_v = 0
                    num = 1
                if node_v in [node_s, node_t]:
                    num_v = 0
                    num = 1
                if node_v != node_s and node_v != node_t and node_s != node_t:
                    num = flow_dict[node_s][node_t]
                    num_v = min(flow_dict[node_s][node_v], flow_dict[node_v][node_t])
                if num == 0:
                    pass
                else:
                    result_dict[node_v] = result_dict[node_v] + num_v / num
    return result_dict


# flow betweenness
def NumberOfFlow(G):
    nodes = G.nodes
    result_dict = dict()
    for node1, _ in nodes.items():
        result_dict[node1] = dict()
        for node2, _ in nodes.items():
            if node1 == node2:
                pass
            else:
                result_dict[node1][node2] = edmonds_karp(G, node1, node2)
    return result_dict


def edmonds_karp(G, source, sink):
    nodes = G.nodes
    parent = dict()
    for node, _ in nodes.items():
        parent[node] = -1

    adj = copy.deepcopy(G.adj)
    max_flow = 0
    while bfs(G, source, sink, parent, adj):
        path_flow = float("inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, adj[parent[s]][s].get("weight", 1))
            s = parent[s]
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            x = adj[u][v].get("weight", 1)
            adj[u][v].update({"weight": x})
            adj[u][v]["weight"] -= path_flow

            flag = 0
            if v not in adj:
                adj[v] = dict()
            if u not in adj[v]:
                adj[v][u] = dict()
                flag = 1
            if flag == 1:
                x = 0
            else:
                x = adj[v][u].get("weight", 1)
            adj[v][u].update({"weight": x})
            adj[v][u]["weight"] += path_flow
            v = parent[v]
    return max_flow


def bfs(G, source, sink, parent, adj):
    nodes = G.nodes
    visited = dict()
    for node, _ in nodes.items():
        visited[node] = 0
    queue = collections.deque()
    queue.append(source)
    visited[source] = True
    while queue:
        u = queue.popleft()
        if u not in adj:
            continue
        for v, attr in adj[u].items():
            if (visited[v] == False) and (attr.get("weight", 1) > 0):
                queue.append(v)
                visited[v] = True
                parent[v] = u
    return visited[sink]
