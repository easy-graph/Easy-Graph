__all__=[
    "Dijkstra",
    "Floyd",
    "Prim",
    "Kruskal"
]

def Dijkstra(G,node):
    adj=G.adj.copy()
    n=len(adj)+1
    visited=[0]*n
    result_dict={}
    temp_key = adj[node].keys()
    for i in range(1,n):
        if i in temp_key:
            result_dict[i]=adj[node][i]['weight']
        else:
            result_dict[i]=float("inf") 
    result_dict[node]=0
    visited[node]=1
    for i in range(1,n): 
        min=float("inf") 
        for j in range(1,n):
            if not visited[j] and result_dict[j] < min:
                k = j
                min = result_dict[j]
        visited[k] = 1
        for j in range(1,n):
            if not visited[j] and j in adj[k].keys() and min + adj[k][j]['weight'] < result_dict[j]:
                result_dict[j] = min + adj[k][j]['weight']
    return result_dict

def Floyd(G):
    adj=G.adj.copy()
    n=len(adj)+1
    result_dict={}
    for i in range(1,n):
        result_dict[i]={}
    for i in range(1,n):
        temp_key = adj[i].keys()
        for j in range(1,n):
            if j in temp_key:
                result_dict[i][j]=adj[i][j]['weight']
            else:
                result_dict[i][j]=float("inf") 
            if i==j:
                result_dict[i][i]=0
    for k in range(1,n):
        for i in range(1,n): 
            for j in range(1,n):
                temp = result_dict[i][k] + result_dict[k][j]  
                if result_dict[i][j] > temp:  
                    result_dict[i][j] = temp  
    return result_dict

def Prim(G):
    adj=G.adj.copy()
    n=len(adj)+1
    result_dict={}
    for i in range(1,n):
        result_dict[i]={}
    selected=[1]
    candidate=[i for i in range(2, n)]
    while len(candidate):
        start=0
        end=0
        min_weight=float("inf")
        for i in selected:
            for j in candidate:
                if i in adj.keys() and j in adj[i].keys() and adj[i][j]['weight']<min_weight:
                    start=i
                    end=j
                    min_weight=adj[i][j]['weight']
        result_dict[start][end]=min_weight
        selected.append(end)
        candidate.remove(end)
    return result_dict


def Kruskal(G):
    adj=G.adj.copy()
    n=len(adj)+1
    result_dict={}
    edge_list=[]
    for i in range(1,n):
        result_dict[i]={}
    for i in range(1,n):
        if i in adj.keys():
            for j in range(1,n):
                if j in adj[i].keys():
                    weight=adj[i][j]['weight']
                    edge_list.append([i,j,weight])
    edge_list.sort(key=lambda a:a[2])
    group = [[i] for i in range(1,n)]
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

