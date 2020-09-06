__all__=[
    "Dijkstra",
    "Floyd",
    "Prim",
    "Kruskal"
]

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
