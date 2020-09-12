import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
import statsmodels.api as sm
import easygraph as eg 

__all__=[
    "plot_Followers",
    "plot_Connected_Communities",
    "plot_Betweenness_Centrality",
    "plot_Neighborhood_Followers"
]

# Number of Followers
def plot_Followers(G,sh):
    ou=[]
    for i in G:
        if i not in sh:
            ou.append(i)
    degree=G.degree()
    sample1=[]
    sample2=[]
    for i in degree.keys():
        if i in ou:
            sample1.append(degree[i])
        elif i in sh:
            sample2.append(degree[i])
    X1=np.linspace(0, max(sample1))
    ecdf = sm.distributions.ECDF(sample1)
    Y1=ecdf(X1)
    X2=np.linspace(0, max(sample2))
    ecdf = sm.distributions.ECDF(sample2)
    Y2=ecdf(X2)
    plt.plot(X1,Y1,'b--',label='Ordinary User')
    plt.plot(X2,Y2,'r',label='SH Spanner')
    plt.title('Number of Followers')
    plt.xlabel('Number of Followers')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend(loc='lower right')

# Number of Connected Communities
def plot_Connected_Communities(G,sh):
    ou=[]
    for i in G:
        if i not in sh:
            ou.append(i)
    sample1=[]
    sample2=[]
    cmts=eg.greedy_modularity_communities(G)
    for i in ou:
        k=0
        for j in range(0,len(cmts)):
            if i in cmts[j]:
                k=k+1
        sample1.append(k)
    for i in sh:
        k=0
        for j in range(0,len(cmts)):
            if i in cmts[j]:
                k=k+1
        sample2.append(k)
    X1=np.linspace(0, max(sample1))
    ecdf = sm.distributions.ECDF(sample1)
    Y1=ecdf(X1)
    X2=np.linspace(0, max(sample2))
    ecdf = sm.distributions.ECDF(sample2)
    Y2=ecdf(X2)
    plt.plot(X1,Y1,'b--',label='Ordinary User')
    plt.plot(X2,Y2,'r',label='SH Spanner')
    plt.title('Number of Connected Communities')
    plt.xlabel('Number of Connected Communities')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend(loc='lower right')

# Betweenness Centrality
def plot_Betweenness_Centrality(G,sh):
    ou=[]
    for i in G:
        if i not in sh:
            ou.append(i)
    bc=eg.betweenness_centrality(G)
    sample1=[]
    sample2=[]
    for i in bc.keys():
        if i in ou:
            sample1.append(bc[i])
        elif i in sh:
            sample2.append(bc[i])
    X1=np.linspace(0, max(sample1))
    ecdf = sm.distributions.ECDF(sample1)
    Y1=ecdf(X1)
    X2=np.linspace(0, max(sample2))
    ecdf = sm.distributions.ECDF(sample2)
    Y2=ecdf(X2)
    plt.plot(X1,Y1,'b--',label='Ordinary User')
    plt.plot(X2,Y2,'r',label='SH Spanner')
    plt.title('Betweenness Centrality')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend(loc='lower right')

# Arg. Number of Followers of the Neighborhood Users
def plot_Neighborhood_Followers(G,sh):
    ou=[]
    for i in G:
        if i not in sh:
            ou.append(i)
    sample1=[]
    sample2=[]
    degree=G.degree()
    for i in ou:
        num=0
        sum=0
        for neighbor in G.neighbors(node=i):
            num=num+1
            sum=sum+degree[neighbor]
        sample1.append(sum/num)
    for i in sh:
        num=0
        sum=0
        for neighbor in G.neighbors(node=i):
            num=num+1
            sum=sum+degree[neighbor]
        sample2.append(sum/num)
    X1=np.linspace(0, max(sample1))
    ecdf = sm.distributions.ECDF(sample1)
    Y1=ecdf(X1)
    X2=np.linspace(0, max(sample2))
    ecdf = sm.distributions.ECDF(sample2)
    Y2=ecdf(X2)
    plt.plot(X1,Y1,'b--',label='Ordinary User')
    plt.plot(X2,Y2,'r',label='SH Spanner')
    plt.title('Arg. Number of Followers of the Neighborhood Users')
    plt.xlabel('Arg. Number of Followers of the Neighborhood Users')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend(loc='lower right')
    plt.show()