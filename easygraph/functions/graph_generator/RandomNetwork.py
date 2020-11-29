import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import random
import time
 
__all__ = [
    'ER_M',
    'ER_P',
    'WS',
]

def ER_M(NETWORK_SIZE,EDGE_NUMBER):
    G=eg.Graph()
    adjacentMatrix=np.zeros((NETWORK_SIZE,NETWORK_SIZE),dtype=int)
    count=0
    while count<EDGE_NUMBER:
        i=random.randint(0,NETWORK_SIZE-1)
        j=random.randint(0,NETWORK_SIZE-1)
        if adjacentMatrix[i][j]==0 and i!=j:
            count =count+1
            adjacentMatrix[i][j]=adjacentMatrix[j][i]=1
            G.add_edge(i,j)
    writeRandomNetworkToFile(NETWORK_SIZE,adjacentMatrix)
    return G

def ER_P(NETWORK_SIZE,PROBABILITY):
    G = eg.Graph()
    adjacentMatrix=np.zeros((NETWORK_SIZE,NETWORK_SIZE),dtype=int)
    count=0
    probability=0.0
    for i in range(NETWORK_SIZE):
        for j in range(i+1,NETWORK_SIZE):
            probability=random.random()
            if probability<PROBABILITY:
                count =count+1
                adjacentMatrix[i][j]=adjacentMatrix[j][i]=1
                G.add_edge(i,j)
    writeRandomNetworkToFile(NETWORK_SIZE,adjacentMatrix)
    return G

def WS(n,k,p):
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
    
def writeRandomNetworkToFile(NETWORK_SIZE,adjacentMatrix):
    f=open('randomNetwork.txt','w+')
    for i in range(NETWORK_SIZE):
        for j in range(NETWORK_SIZE):
            if adjacentMatrix[i][j]==1:
                f.write(str(i))
                f.write(' ')
                f.write(str(j))
                f.write('\n')
    f.close()



