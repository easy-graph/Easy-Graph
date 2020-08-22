__all__=[
    "ego_betweenness"
]
import numpy as np 
import numpy.matlib 

def ego_betweenness(G,node):
    """Returns the betweenness centrality of a ego network whose ego is set

    Parameters
    ---------- 
    G : graph
    node : int
    
    Returns
    -------
    sum : float
        the betweenness centrality of a ego network whose ego is set

    Examples
    --------
    Returns the betwenness centrality of node 1.

    >>> ego_betweenness(G,node=1)

    Reference
    ---------
    .. [1] Martin Everett, Stephen P. Borgatti. "Ego network betweenness."
    Social Networks, Volume 27, Issue 1, Pages 31-38, 2005.

    """
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

