import collections
import copy
import sys
sys.path.append('../../../')
from easygraph.functions.path import *
__all__ = [
    "betweenness_centrality",
    "closeness_centrality",
    "flowbetweenness_centrality"
]
def closeness_centrality(G):
    '''Compute closeness centrality for nodes.

    .. math::

        C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},
    
    Notice that the closeness distance function computes the 
    outcoming distance to `u` for directed graphs. To use 
    incoming distance, act on `G.reverse()`.

    Parameters
    ----------
    G : graph
      A easygraph graph

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    '''
    result_dict = dict()
    length = len(G.nodes)
    distance_dict = GetAllDistance(G)
    for node1,x in distance_dict.items():
        dist = 0
        cnt = 0
        for _,distance in x.items():
            if distance == float("inf"):
                pass
            else:
                cnt = cnt+1
                dist = distance+dist
        if dist  == 0:
            result_dict[node1] = 0
        else:
            result_dict[node1] = (cnt-1)*(cnt-1)/(dist*(length-1))
    return result_dict

def betweenness_centrality(G):
    '''Compute the shortest-path betweenness centrality for nodes.

    .. math::

        c_B(v)  = \sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths,  and $\sigma(s, t|v)$ is the number of
    those paths  passing through some  node $v$ other than $s, t$.
    If $s = t$, $\sigma(s, t) = 1$, and if $v \in {s, t}$,
    $\sigma(s, t|v) = 0$ [2]_.

    Parameters
    ----------
    G : graph
      A easygraph graph.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.
    '''
    distance_dict = GetAllDistance(G)
    Shortest_dict = NumberOfShortest(G,distance_dict)
    nodes = G.nodes
    result_dict = dict()
    for node,_ in nodes.items():
        result_dict[node] = 0
    for node_v,_ in nodes.items():
        for node_s,_ in nodes.items():
            for node_t,_ in nodes.items():
                num = 1
                num_v = 0
                if node_s == node_t:
                    num_v = 0
                    num = 1
                if node_v in [node_s,node_t]:
                    num_v = 0
                    num = 1
                if node_v != node_s and node_v != node_t and node_s != node_t:
                    num = Shortest_dict[node_s][node_t]
                    if distance_dict[node_s][node_t] == (distance_dict[node_s][node_v]+distance_dict[node_v][node_t]):
                        num_v = Shortest_dict[node_s][node_v]*Shortest_dict[node_v][node_t]
                if num == 0:
                    pass
                else:
                    result_dict[node_v] = result_dict[node_v]+num_v/num
    return result_dict

def flowbetweenness_centrality(G):
    '''Compute the independent-path betweenness centrality for nodes in a flow network.

    .. math::

       c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    independent $(s, t)$-paths,  and $\sigma(s, t|v)$ is the maximum 
    number possible of those paths  passing through some  node $v$ 
    other than $s, t$. If $s = t$, $\sigma(s, t) = 1$, and if $v \in {s, t}$,
    $\sigma(s, t|v) = 0$ [2]_.

    Parameters
    ----------
    G : graph
      A easygraph directed graph.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with independent-path betweenness centrality as the value.

    Notes
    -----
    A flow network is a directed graph where each edge has a capacity and each edge receives a flow. 
    '''
    if G.is_directed() == False:
        print("Please input a directed graph")
        return 
    flow_dict = NumberOfFlow(G)
    nodes = G.nodes
    result_dict = dict()
    for node,_ in nodes.items():
        result_dict[node] = 0
    for node_v,_ in nodes.items():
        for node_s,_ in nodes.items():
            for node_t,_ in nodes.items():
                num = 1
                num_v = 0
                if node_s  == node_t:
                    num_v = 0
                    num = 1
                if node_v in [node_s,node_t]:
                    num_v = 0
                    num = 1
                if node_v != node_s and node_v != node_t and node_s != node_t:
                    num = flow_dict[node_s][node_t]
                    num_v = min(flow_dict[node_s][node_v],flow_dict[node_v][node_t])
                if num == 0:
                    pass
                else:
                    result_dict[node_v] = result_dict[node_v]+num_v/num
    return result_dict


def GetAllDistance(G):
    result_dict = dict()
    nodes = G.nodes
    #result_dict = Floyd(G) # n^3
    for _, _, d in G.edges:
        if d.get("weight",1) < 0:
            print("Please input a grapg without negative-weight edges !")
            exit()
    for node,_ in nodes.items():
        result_dict[node] = Dijkstra(G,node) # n*(n*n) or n*(n*logn+m)
        #result_dict[node] = Bellman_Ford(G,node) # n*(n*m)
    return result_dict


def Bellman_Ford(G,start_node):
    dis = dict()
    node_list = []
    for node,_ in G.nodes.items():
        node_list.append(node)
    for i in node_list:
        dis[i] = float("inf")
    dis[start_node] = 0
    for i in range(len(node_list)):
        for u,v,w in G.edges().items():
            dis[v] = min(dis[v],dis[u]+w.get("weight",1))
    for u,v,w in G.edges().items():
        if dis[v]>(dis[u]+w.get("weight",1)):
            print("Error: a negative-weight cycle exists!")
            break
    return dis


def NumberOfShortest(G,dis):
    result_dict = dict()
    nodes = G.nodes
    for start_node in nodes:
        passedges = []
        result_dict[start_node] = dict()
        vis = dict()
        count = dict()
        for node in nodes:
            vis[node] = 0
            count[node] = 0
        count[start_node] = 1
        DFS(G,start_node,vis,count,dis[start_node])
        for node in nodes:
            result_dict[start_node][node] = count[node]   
    return result_dict

def DFS(G,u,vis,count,dis):
    vis[u] = 1
    for v,_ in G.adj[u].items():
        if dis[v] == ( dis[u]+G.adj[u][v].get("weight",1)):
            count[v] = count[v]+1
            DFS(G,v,vis,count,dis)

# flow betweenness
def NumberOfFlow(G):
    nodes = G.nodes
    result_dict = dict()
    for node1,_ in nodes.items():
        result_dict[node1] = dict()
        for node2,_ in nodes.items():
            if node1 == node2:
                pass
            else:
                result_dict[node1][node2] = edmonds_karp(G,node1,node2)
    return result_dict

def edmonds_karp(G,source,sink):
    nodes = G.nodes
    parent = dict()
    for node,_ in nodes.items():
        parent[node] = -1
      
    adj = copy.deepcopy(G.adj) 
    max_flow = 0
    while bfs(G,source,sink,parent,adj):
        path_flow = float("inf")
        s = sink
        while s != source:
            path_flow = min(path_flow,adj[parent[s]][s].get("weight",1))
            s = parent[s]
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            x = adj[u][v].get("weight",1)
            adj[u][v].update({"weight":x})
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
                x  = adj[v][u].get("weight",1)
            adj[v][u].update({"weight":x})
            adj[v][u]["weight"] += path_flow
            v = parent[v]
    return max_flow

def bfs(G,source,sink,parent,adj):
    nodes = G.nodes
    visited = dict()
    for node,_ in nodes.items():
        visited[node] = 0
    queue = collections.deque()
    queue.append(source)
    visited[source] = True
    while queue:
        u = queue.popleft()
        if u not in adj:
            continue
        for v, attr in adj[u].items():
            if (visited[v] == False) and (attr.get("weight",1) > 0):
                queue.append(v)
                visited[v] = True
                parent[v] = u
    return visited[sink]
