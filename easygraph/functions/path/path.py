__all__=[
    "weight_function",
    "Dijkstra",
    "Floyd",
    "Prim",
    "Kruskal",
    "Dijkstra_mutilsource_path_predecessor_distance",
    "all_shortest_paths",
    "all_pairs_dijkstra_path"
]

def weight_function(G, weight):
    if callable(weight):
        return weight
    return lambda u, v, data: data.get(weight, 1)


def Dijkstra(G,node):
    """Returns the length of paths from the certain node to remaining nodes

    Parameters
    ---------- 
    G : graph
        weighted graph
    node : int

    Returns
    -------
    result_dict : dict
        the length of paths from the certain node to remaining nodes

    Examples
    --------
    Returns the length of paths from node 1 to remaining nodes

    >>> Dijkstra(G,node=1)

    """
    adj=G.adj.copy()
    visited={}
    result_dict={}
    temp_key = adj[node].keys()
    for i in G:
        if i in temp_key:
            result_dict[i]=adj[node][i].get("weight",1)
        else:
            result_dict[i]=float("inf") 
        visited[i]=0
    result_dict[node]=0
    visited[node]=1
    for i in G: 
        min=float("inf") 
        k = node
        for j in G:
            if not visited[j] and result_dict[j] < min:
                k = j
                min = result_dict[j]
        visited[k] = 1
        for j in G:
            if not visited[j] and j in adj[k].keys() and min + adj[k][j].get("weight",1) < result_dict[j]:
                result_dict[j] = min + adj[k][j].get("weight",1)
    return result_dict

def Dijkstra_mutilsource_path_predecessor_distance(G, sources, weight=None):
    """Returns the length of paths from the certain node to remaining nodes

    Parameters
    ---------- 
    G : graph
        weighted graph
    node : int

    Returns
    -------
    result_dict : dict
        the length of paths from the certain node to remaining nodes

    Examples
    --------
    Returns the length of paths from node 1 to remaining nodes

    >>> Dijkstra(G,node=1)

    """
    adj = G.adj
    pred = {}
    path = {source: [source] for source in sources}
    source = next(iter(sources))
    for i in sources:
        pred[i] = []
    result_dict = {}
    seen_dict = {}
    for i in G:
        if i in sources:
            result_dict[i] = 0
        else:
            result_dict[i] = float("inf")
    for i in G:
        for j in sources:
            if i in adj[j] and i not in sources:
                result_dict[i]=min( result_dict[i],adj[j][i].get("weight",1))
                pred[i] = [j]
                path[i] = path[j] + [i]
    seen_dict[source] = 0
    for i in G: 
        Min=float("inf") 
        k = source
        for j in G:
            if j not in seen_dict and result_dict[j] < Min:
                k = j
                Min = result_dict[j]
        if k not in pred:
            pred[k] = [source]
        seen_dict[k] = Min
        for j in G:
            if j in adj[k]:
                cost = weight(k,j,adj[k][j]) + Min
                if j in seen_dict:
                    if cost < seen_dict[j]:
                        raise ValueError('Contradictory paths found:',
                                        'negative weights?')
                elif cost < result_dict[j]:
                    result_dict[j] = Min + adj[k][j].get("weight",1)
                    pred[j] = [k]
                    path[j] = path[k] + [j]
                elif cost == result_dict[j]:
                    pred[j].append(k) 
    return (path, pred, result_dict)

def all_pairs_dijkstra_path(G, weight=None):
    """Compute shortest paths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    distance : dictionary
       Dictionary, keyed by source and target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = dict(nx.all_pairs_dijkstra_path(G))
    >>> print(path[0][4])
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    floyd_warshall(), all_pairs_bellman_ford_path()

    """
    # TODO This can be trivially parallelized.
    weight = weight_function(G,weight)
    for n in G:
        path, _, _ =  Dijkstra_mutilsource_path_predecessor_distance(G, [n], weight=weight)
        yield (n, path)

def all_shortest_paths(G, source, target, weight=None):
    """Compute all shortest paths in the graph.

    Parameters
    ----------
    G : easygraph graph

    source : node
       Starting node for path.

    target : node
       Ending node for path.

    weight : None or string, optional (default = None)
       If None, every edge has weight/distance/cost 1.
       If a string, use this edge attribute as the edge weight.
       Any edge attribute not present defaults to 1.

    Returns
    -------
    paths : generator of lists
        A generator of all paths between source and target.

    Notes
    -----
    There may be many shortest paths between the source and target.

    """
    weight = weight_function(G,weight)
    _, pred, _ = Dijkstra_mutilsource_path_predecessor_distance(G, [source], weight=weight)  
    if target not in pred:
        print('Target {} cannot be reached'
                                'from Source {}'.format(target, source))
        exit()

    stack = [[target, 0]]
    top = 0
    while top >= 0:
        node, i = stack[top]
        if node == source:
            yield [p for p, n in reversed(stack[:top + 1])]
        if len(pred[node]) > i:
            top += 1
            if top == len(stack):
                stack.append([pred[node][i], 0])
            else:
                stack[top] = [pred[node][i], 0]
        else:
            stack[top - 1][1] += 1
            top -= 1

def Floyd(G):
    """Returns the length of paths from all nodes to remaining nodes

    Parameters
    ---------- 
    G : graph
        weighted graph
    
    Returns
    -------
    result_dict : dict
        the length of paths from all nodes to remaining nodes

    Examples
    --------
    Returns the length of paths from all nodes to remaining nodes

    >>> Floyd(G)

    """
    adj=G.adj.copy()
    result_dict={}
    for i in G:
        result_dict[i]={}
    for i in G:
        temp_key = adj[i].keys()
        for j in G:
            if j in temp_key:
                result_dict[i][j]=adj[i][j].get("weight",1)
            else:
                result_dict[i][j]=float("inf") 
            if i==j:
                result_dict[i][i]=0
    for k in G:
        for i in G: 
            for j in G:
                temp = result_dict[i][k] + result_dict[k][j]  
                if result_dict[i][j] > temp:  
                    result_dict[i][j] = temp  
    return result_dict

def Prim(G):
    """Returns the edges that make up the minimum spanning tree

    Parameters
    ---------- 
    G : graph
        weighted graph
    
    Returns
    -------
    result_dict : dict
        the edges that make up the minimum spanning tree

    Examples
    --------
    Returns the edges that make up the minimum spanning tree

    >>> Prim(G)

    """
    adj=G.adj.copy()
    result_dict={}
    for i in G:
        result_dict[i]={}
    selected=[]
    candidate=[]
    for i in G:
        if not selected:
            selected.append(i)
        else:
            candidate.append(i)
    while len(candidate):
        start=None
        end=None
        min_weight=float("inf")
        for i in selected:
            for j in candidate:
                if i in G and j in G[i] and adj[i][j].get("weight",1)<min_weight:
                    start=i
                    end=j
                    min_weight=adj[i][j].get("weight",1)
        if start!=None and end!=None:
            result_dict[start][end]=min_weight
            selected.append(end)
            candidate.remove(end)
        else:
            break
    return result_dict

def Kruskal(G):
    """Returns the edges that make up the minimum spanning tree

    Parameters
    ---------- 
    G : graph
        weighted graph
    
    Returns
    -------
    result_dict : dict
        the edges that make up the minimum spanning tree

    Examples
    --------
    Returns the edges that make up the minimum spanning tree

    >>> Kruskal(G)

    """
    adj=G.adj.copy()
    result_dict={}
    edge_list=[]
    for i in G:
        result_dict[i]={}
    for i in G:
        for j in G[i]:
            weight=adj[i][j].get("weight",1)
            edge_list.append([i,j,weight])
    edge_list.sort(key=lambda a:a[2])
    group = [[i] for i in G]
    for edge in edge_list:
      for i in range(len(group)):
        if edge[0] in group[i]:
          m = i
        if edge[1] in group[i]:
          n = i
      if m != n:
        result_dict[edge[0]][edge[1]]=edge[2]
        group[m] = group[m] + group[n]
        group[n] = []
    return result_dict
