def  laplacian(G):
    """Returns the laplacian centrality of each node in the weighted graph

    Parameters
    ---------- 
    G : graph
        weighted graph
    
    Returns
    -------
    CL : dict
        the laplacian centrality of each node in the weighted graph

    Examples
    --------
    Returns the laplacian centrality of each node in the weighted graph G

    >>> laplacian(G)

    Reference
    ---------
    .. [1] Xingqin Qi, Eddie Fuller, Qin Wu, Yezhou Wu, Cun-Quan Zhang. 
    "Laplacian centrality: A new centrality measure for weighted networks." 
    Information Sciences, Volume 194, Pages 240-253, 2012.

    """
    adj=G.adj
    X={}
    W={}
    CL={}
    Xi={}
    for i in G:
        X[i]=0
        W[i]=0
        CL[i]=0
        Xi[i]=0
    for i in G:
        for j in G:
            if i in G and j in G[i]:
                X[i]+=adj[i][j]['weight']
                W[i]+=adj[i][j]['weight']*adj[i][j]['weight']
    ELG=sum(X[i]*X[i] for i in G)+sum(W[i] for i in G)
    for i in G:
        Xi=X
        for j in G:
            if j in adj.keys() and i in adj[j].keys():
                Xi[j]-=adj[j][i]['weight']
        Xi[i]=0
        ELGi=sum(Xi[i]*Xi[i] for i in G)+sum(W[i] for i in G)-2*W[i]
        if ELG:
            CL[i]=(float)(ELG-ELGi)/ELG
    return CL