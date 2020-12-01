import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import random
import time
 
__all__ = [
    'erdos_renyi_M',
    'erdos_renyi_P',
    'WS_Random',
]

def erdos_renyi_M(n,edge,FilePath=None):
    """Given the number of nodes and the number of edges, return an Erdős-Rényi random graph, and store the graph in a document.

    Parameters
    ----------
    n : int
        The number of nodes.
    edge : int
        The number of edges.
    FilePath : string
        The file path of storing the graph G. 

    Returns
    -------
    G : graph
        an Erdős-Rényi random graph.

    Examples
    --------
    Returns an Erdős-Rényi random graph G.

    >>> erdos_renyi_M(100,180,"/users/fudanmsn/downloads/RandomNetwork.txt")

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    G=eg.Graph()
    adjacentMatrix=np.zeros((n,n),dtype=int)
    count=0
    while count<edge:
        i=random.randint(0,n-1)
        j=random.randint(0,n-1)
        if adjacentMatrix[i][j]==0 and i!=j:
            count =count+1
            adjacentMatrix[i][j]=adjacentMatrix[j][i]=1
            G.add_edge(i,j)
    writeRandomNetworkToFile(n,adjacentMatrix,FilePath)
    return G

def erdos_renyi_P(n,p,FilePath=None):
    """Given the number of nodes and the probability of edge creation, return an Erdős-Rényi random graph, and store the graph in a document.
    
    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    FilePath : string
        The file path of storing the graph G.

    Returns
    -------
    G : graph
        an Erdős-Rényi random graph.

    Examples
    --------
    Returns an Erdős-Rényi random graph G

    >>> erdos_renyi_P(100,0.5,"/users/fudanmsn/downloads/RandomNetwork.txt")

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    G = eg.Graph()
    adjacentMatrix=np.zeros((n,n),dtype=int)
    count=0
    probability=0.0
    for i in range(n):
        for j in range(i+1,n):
            probability=random.random()
            if probability<p:
                count =count+1
                adjacentMatrix[i][j]=adjacentMatrix[j][i]=1
                G.add_edge(i,j)
    writeRandomNetworkToFile(n,adjacentMatrix,FilePath)
    return G

def WS_Random(n,k,p,FilePath=None):
    """Returns a small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    FilePath : string
        The file path of storing the graph G 

    Returns
    -------
    G : graph
        a small-world graph

    Examples
    --------
    Returns a small-world graph G

    >>> WS_Random(100,10,0.3,"/users/fudanmsn/downloads/RandomNetwork.txt")

    """
    adjacentMatrix=np.zeros((n,n),dtype=int)
    G = eg.Graph()
    NUM1 = n
    NUM2 = NUM1 - 1
    K = k          
    K1 = K + 1
    g = eg.Graph()
    N = list(range(NUM1))
    g.add_nodes(N)   
 
    for i in range(NUM1):
        for j in range(1, K1):
            K_add = NUM1 - K
            i_add_j = i + j + 1
            if i >= K_add and i_add_j > NUM1:  
                i_add = i + j - NUM1
                g.add_edge(i, i_add)
                adjacentMatrix[i][i_add]=adjacentMatrix[i_add][i]=1
            else:
                i_add = i + j
                g.add_edge(i, i_add)
                adjacentMatrix[i][i_add]=adjacentMatrix[i_add][i]=1
 
    for i in range(NUM1):
        for e_del in range(i + 1, i + K1):
            if e_del >= NUM1:     
                e_del = e_del - NUM1
            P_random = random.randint(0, 9)
            if P_random <= p*10-1:
                g.remove_edge(i, e_del)
                adjacentMatrix[i][e_del]=adjacentMatrix[e_del][i]=0
                e_add = random.randint(0, NUM2)    
                while e_add == i or g.has_edge(i, e_add) == True:
                    e_add = random.randint(0, NUM2)
                g.add_edge(i, e_add)
                adjacentMatrix[i][e_add]=adjacentMatrix[e_add][i]=1
    writeRandomNetworkToFile(n,adjacentMatrix)
    return G
    
def writeRandomNetworkToFile(n,adjacentMatrix,FilePath):
    if FilePath!=None:
        f=open(FilePath,'w+')
    else:
        f=open("RandomNetwork.txt",'w+')
    for i in range(n):
        for j in range(n):
            if adjacentMatrix[i][j]==1:
                f.write(str(i))
                f.write(' ')
                f.write(str(j))
                f.write('\n')
    f.close()



