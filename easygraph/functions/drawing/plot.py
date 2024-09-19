import easygraph as eg


__all__ = [
    "plot_Followers",
    "plot_Connected_Communities",
    "plot_Betweenness_Centrality",
    "plot_Neighborhood_Followers",
]


# Number of Followers
def plot_Followers(G, SHS):
    """
    Returns the CDF curves of "Number of Followers" of SH spanners and ordinary users in graph G.

    Parameters
    ----------
    G : graph
        A easygraph graph.

    SHS : list
        The SH Spanners in graph G.

    Returns
    -------
    plt : CDF curves
        the CDF curves of "Number of Followers" of SH spanners and ordinary users in graph G.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sm

    assert len(SHS) < len(
        G.nodes
    ), "The number of SHS must be less than the number of nodes in the graph."
    OU = []
    for i in G:
        if i not in SHS:
            OU.append(i)
    degree = G.degree()
    sample1 = []
    sample2 = []
    for i in degree.keys():
        if i in OU:
            sample1.append(degree[i])
        elif i in SHS:
            sample2.append(degree[i])
    X1 = np.linspace(min(sample1), max(sample1))
    ecdf = sm.distributions.ECDF(sample1)
    Y1 = ecdf(X1)
    X2 = np.linspace(min(sample2), max(sample2))
    ecdf = sm.distributions.ECDF(sample2)
    Y2 = ecdf(X2)
    plt.plot(X1, Y1, "b--", label="Ordinary User")
    plt.plot(X2, Y2, "r", label="SH Spanner")
    plt.title("Number of Followers")
    plt.xlabel("Number of Followers")
    plt.ylabel("Cumulative Distribution Function")
    plt.legend(loc="lower right")
    plt.show()


# Number of Connected Communities
def plot_Connected_Communities(G, SHS):
    """
    Returns the CDF curves of "Number of Connected Communities" of SH spanners and ordinary users in graph G.

    Parameters
    ----------
    G : graph
        A easygraph graph.

    SHS : list
        The SH Spanners in graph G.

    Returns
    -------
    plt : CDF curves
        the CDF curves of "Number of Connected Communities" of SH spanners and ordinary users in graph G.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sm

    OU = []
    for i in G:
        if i not in SHS:
            OU.append(i)
    sample1 = []
    sample2 = []
    cmts = eg.LPA(G)
    for i in OU:
        s = set()
        neighbors = G.neighbors(node=i)
        for j in neighbors:
            for k in cmts:
                if j in cmts[k]:
                    s.add(k)
        sample1.append(len(s))
    for i in SHS:
        s = set()
        neighbors = G.neighbors(node=i)
        for j in neighbors:
            for k in cmts:
                if j in cmts[k]:
                    s.add(k)
        sample2.append(len(s))
    print(len(cmts))
    print(sample1)
    print(sample2)
    X1 = np.linspace(min(sample1), max(sample1))
    ecdf = sm.distributions.ECDF(sample1)
    Y1 = ecdf(X1)
    X2 = np.linspace(min(sample2), max(sample2))
    ecdf = sm.distributions.ECDF(sample2)
    Y2 = ecdf(X2)
    plt.plot(X1, Y1, "b--", label="Ordinary User")
    plt.plot(X2, Y2, "r", label="SH Spanner")
    plt.title("Number of Connected Communities")
    plt.xlabel("Number of Connected Communities")
    plt.ylabel("Cumulative Distribution Function")
    plt.legend(loc="lower right")
    plt.show()


# Betweenness Centrality
def plot_Betweenness_Centrality(G, SHS):
    """
    Returns the CDF curves of "Betweenness Centralitys" of SH spanners and ordinary users in graph G.

    Parameters
    ----------
    G : graph
        A easygraph graph.

    SHS : list
        The SH Spanners in graph G.

    Returns
    -------
    plt : CDF curves
        the CDF curves of "Betweenness Centrality" of SH spanners and ordinary users in graph G.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sm

    OU = []
    for i in G:
        if i not in SHS:
            OU.append(i)
    bc = eg.betweenness_centrality(G)
    bc = dict(zip(G.nodes, bc))
    sample1 = []
    sample2 = []
    for i in bc.keys():
        if i in OU:
            sample1.append(bc[i])
        else:
            sample2.append(bc[i])
    X1 = np.linspace(min(sample1), max(sample1))
    ecdf = sm.distributions.ECDF(sample1)
    Y1 = ecdf(X1)
    X2 = np.linspace(min(sample2), max(sample2))
    ecdf = sm.distributions.ECDF(sample2)
    Y2 = ecdf(X2)
    plt.plot(X1, Y1, "b--", label="Ordinary User")
    plt.plot(X2, Y2, "r", label="SH Spanner")
    plt.title("Betweenness Centrality")
    plt.xlabel("Betweenness Centrality")
    plt.ylabel("Cumulative Distribution Function")
    plt.legend(loc="lower right")
    plt.show()


# Arg. Number of Followers of the Neighborhood Users
def plot_Neighborhood_Followers(G, SHS):
    """
    Returns the CDF curves of "Arg. Number of Followers of the Neighborhood Users" of SH spanners and ordinary users in graph G.

    Parameters
    ----------
    G : graph
        A easygraph graph.

    SHS : list
        The SH Spanners in graph G.

    Returns
    -------
    plt : CDF curves
        the CDF curves of "Arg. Number of Followers of the Neighborhood Users
        " of SH spanners and ordinary users in graph G.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sm

    OU = []
    for i in G:
        if i not in SHS:
            OU.append(i)
    sample1 = []
    sample2 = []
    degree = G.degree()
    for i in OU:
        num = 0
        sum = 0
        for neighbor in G.neighbors(node=i):
            num = num + 1
            sum = sum + degree[neighbor]
        sample1.append(sum / num)
    for i in SHS:
        num = 0
        sum = 0
        for neighbor in G.neighbors(node=i):
            num = num + 1
            sum = sum + degree[neighbor]
        sample2.append(sum / num)
    X1 = np.linspace(min(sample1), max(sample1))
    ecdf = sm.distributions.ECDF(sample1)
    Y1 = ecdf(X1)
    X2 = np.linspace(min(sample2), max(sample2))
    ecdf = sm.distributions.ECDF(sample2)
    Y2 = ecdf(X2)
    plt.plot(X1, Y1, "b--", label="Ordinary User")
    plt.plot(X2, Y2, "r", label="SH Spanner")
    plt.title("Arg. Number of Followers of the Neighborhood Users")
    plt.xlabel("Arg. Number of Followers of the Neighborhood Users")
    plt.ylabel("Cumulative Distribution Function")
    plt.legend(loc="lower right")
    plt.show()
