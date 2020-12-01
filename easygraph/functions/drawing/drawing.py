import easygraph as eg 
import matplotlib.pyplot as plt
import numpy as np
import random

__all__=[
    "SHS_layout"
]

def SHS_layout(G,SHS):
    """
    Draw the graph whose the SH Spanners are in the center.

    Parameters
    ----------
    G : graph
        A easygraph graph.

    SHS : list
        The SH Spanners in graph G.

    Returns
    -------
    graph : network
        the graph whose the SH Spanners are in the center.
    """
    pos=eg.random_position(G)
    center=np.zeros((len(SHS),2),float) 
    node=np.zeros((len(pos),2),float)
    m,n=0,0
    for i in pos:
        if i in SHS:
            node[m][0]=0.5+(-1)**random.randint(1,2)*pos[i][0]/5
            node[m][1]=0.5+(-1)**random.randint(1,2)*pos[i][1]/5
            center[n][0]=node[m][0]
            center[n][1]=node[m][1]
            pos[i][0]=node[m][0]
            pos[i][1]=node[m][1]
            m+=1
            n+=1
        else:
            node[m][0]=pos[i][0]
            node[m][1]=pos[i][1]
            m+=1
    plt.scatter(node[:,0], node[:,1], marker = '.', color = 'b', s=10)
    plt.scatter(center[:,0], center[:,1], marker = '*', color = 'r', s=20)

    k=0
    for i in pos:
        plt.text(pos[i][0], pos[i][1], i,
        fontsize=5,
        verticalalignment="top",
        horizontalalignment="right")
        k+=1
    
    for i in G.edges:
        p1=[pos[i[0]][0],pos[i[1]][0]]
        p2=[pos[i[0]][1],pos[i[1]][1]]
        plt.plot(p1,p2, 'k--',alpha=0.3) 
    
    plt.show()
