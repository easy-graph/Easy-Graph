__all__=[
    "ego_betweenness"
]
import numpy as np 
import numpy.matlib 

def ego_betweenness(G,node):
    g=G.ego_subgraph(node)
    n=len(g)+1
    A=np.matlib.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if g.has_edge(i,j):
                A[i,j]=1
    B=A*A
    C=1-A
    sum=0
    flag=G.is_directed()
    for i in range(n):
        for j in range(n):
            if i!=j and C[i,j]==1 and B[i,j]!=0:
                sum+=1.0/B[i,j]
    if flag==False:
        sum/=2
    return sum

