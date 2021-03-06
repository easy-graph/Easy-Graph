import numpy as np
import easygraph as eg
import scipy as sp
from scipy.sparse.linalg import eigs

__all__ = [
    "NOBE",
    "NOBE_GA"
]

def NOBE(G,K):
    """Graph embedding via NOBE[1].

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    K : int
        Embedding dimension k

    Returns
    -------
    Y : list
        list of embedding vectors (y1, y2, · · · , yn)

    Examples
    --------
    >>> NOBE(G,K=15)

    References
    ----------
    .. [1] https://www.researchgate.net/publication/325004496_On_Spectral_Graph_Embedding_A_Non-Backtracking_Perspective_and_Graph_Approximation
    
    """
    LG=graph_to_d_atleast2(G)
    N=len(LG)
    P,pair=Transition(LG)
    V=eigs_nodes(P,K,N)
    print(V)
    print(np.shape(V))
    Y=embedding(V,pair,K,N)
    return Y

def NOBE_GA(G,K):
    """Graph embedding via NOBE-GA[1].

    Parameters
    ----------
    G : easygraph.Graph
        An unweighted and undirected graph.

    K : int
        Embedding dimension k

    Returns
    -------
    Y : list
        list of embedding vectors (y1, y2, · · · , yn)

    Examples
    --------
    >>> NOBE_GA(G,K=15)

    References
    ----------
    .. [1] https://www.researchgate.net/publication/325004496_On_Spectral_Graph_Embedding_A_Non-Backtracking_Perspective_and_Graph_Approximation
    
    """
    N=len(G)
    A = np.eye(N,N)
    for i in G.edges:
        (u,v,t)=i
        u=int(u)
        v=int(v)
        A[u,v]=1
    degree=G.degree()
    D_inv=np.zeros([N,N])
    a=0
    for i in degree:
        D_inv[a,a]=1/degree[i]
        a+=1
    D_I_inv=np.zeros([N,N])
    b=0
    for i in degree:
        D_inv[b,b]=1/(degree[i]-1)
        b+=1
    I=np.identity(N)
    M_D = 0.5*A*D_I_inv*(I-D_inv)
    D_D = 0.5*I
    T_ua = np.zeros([2*N,2*N])       
    T_ua[0:N,0:N] = M_D
    T_ua[N:2*N,N:2*N] = M_D
    T_ua[N:2*N,0:N] = D_D
    T_ua[0:N,N:2*N] = D_D
    Y1,Y=eigs(T_ua,K,which='LR')
    Y=Y[0:N,:]
    return Y

def graph_to_d_atleast2(G):
    n=len(G)
    new_node = n
    degree=G.degree()
    node=G.nodes.copy()
    for i in node:
        if degree[i] == 1:
            for neighbors in G.neighbors(node=i):
                G.add_edge(i,new_node)
                G.add_edge(new_node,neighbors)
                break
            new_node =  new_node + 1
    return G

def Transition(LG):
    N=len(LG)
    M=LG.size()
    LLG=eg.DiGraph()
    for i in LG.edges:
        (u,v,t)=i
        LLG.add_edge(u,v)
        LLG.add_edge(v,u)
    degree=LLG.degree()
    P=np.zeros([2*M,2*M])
    pair=np.zeros([2*M,2])
    k=0
    l=0
    for i in LLG.edges:
        l=0
        for j in LLG.edges:
            (u,v,t)=i
            (x,y,z)=j
            if v==x and u!=y:
                P[k][l]=1/(degree[v]-1)
            l+=1
        k+=1
    a=0
    for i in LLG.edges:
        (u,v,t)=i
        pair[a]=[u,v]
        a+=1
    return P,pair

def eigs_nodes(P,K,N):
    M=np.size(P,0)
    L=np.zeros([M,M])
    I=np.identity(M)
    P_T=P.T
    L=I-(P+P_T)/2
    U,D = eigs(L,K+1,which='LR')
    D=D[:,:-1]
    V=np.zeros([M,K],dtype = complex)
    a=0
    for i in D:
        V[a]=i
        a+=1
    return V

def embedding(V,pair,K,N):
    Y=np.zeros([N,K],dtype = complex)
    idx=0
    for i in pair:
        [v,u]=i
        u=int(u)
        for j in range(0, len(V[idx])):
            Y[u,j]+=V[idx,j] 
        idx+=1
    return Y
 