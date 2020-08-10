def  laplacian(G,nodes=None,weight=None):
    adj=G.adj
    n=len(adj)+1
    X=[0]*n
    W=[0]*n
    CL={}
    Xi=[0]*n
    for i in range(1,n):
        for j in range(1,n):
            if i in adj.keys() and j in adj[i].keys():
                X[i]+=adj[i][j]['weight']
                W[i]+=adj[i][j]['weight']*adj[i][j]['weight']
    ELG=sum(X[i]*X[i] for i in range(1,n))+sum(W[i] for i in range(1,n))
    for i in range(1,n):
        Xi=list(X)
        for j in range(1,n):
            if j in adj.keys() and i in adj[j].keys():
                Xi[j]-=adj[j][i]['weight']
        Xi[i]=0
        ELGi=sum(Xi[i]*Xi[i] for i in range(1,n))+sum(W[i] for i in range(1,n))-2*W[i]
        if ELG:
            CL[i]=(float)(ELG-ELGi)/ELG
    return CL