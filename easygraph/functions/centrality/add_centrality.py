import collections
import copy
import sys
import networkx as nx
import easygraph as eg
from easygraph.functions.path import *
from easygraph.functions.centrality.flow_matrix import *
from easygraph.functions.centrality.centrality import *
from .convert_matrix import *
from .rcm import *
from math import sqrt
from itertools import combinations
from functools import partial
from itertools import tee, chain
__all__ = [
    "degree_centrality",
    "current_flow_betweenness_centrality",
    "eigenvector_centrality",
    "current_flow_closeness_centrality",
    "second_order_centrality",
    "communicability_betweenness_centrality",
    "group_betweenness_centrality",
    "group_closeness_centrality",
    "group_degree_centrality",
    "load_centrality",
    "subgraph_centrality",
    "harmonic_centrality",
    "dispersion",
    "global_reaching_centrality"
]

def relabel_nodes(G,ordering,n):
    H=eg.Graph()
    Match = dict(zip(ordering,range(n)))
    for u, v, d in G.edges:
        H.add_edge(Match[u],Match[v],weight = d.get("weight",1))
    return H

def degree_centrality(G):
    """Compute the degree centrality for nodes.

    The degree centrality for a node v is the fraction of nodes it
    is connected to.

    Parameters
    ----------
    G : graph
      A easygraph graph

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with degree centrality as the value.

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    n = len(G)
    if n <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (n - 1.0)
    result_dict = {n: d * s for n, d in G.degree().items()}
    return result_dict

# not_implemented_for("multigraph")
def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, weight=None):
    r"""Compute the eigenvector centrality for the graph `G`.

    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node $i$ is
    the $i$-th element of the vector $x$ defined by the equation

    .. math::

        Ax = \lambda x

    where $A$ is the adjacency matrix of the graph `G` with eigenvalue
    $\lambda$. By virtue of the Perron–Frobenius theorem, there is a unique
    solution $x$, all of whose entries are positive, if $\lambda$ is the
    largest eigenvalue of the adjacency matrix $A$ ([2]_).

    Parameters
    ----------
    G : graph
      A easygraph graph

    max_iter : integer, optional (default=100)
      Maximum number of iterations in power method.

    tol : float, optional (default=1.0e-6)
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional (default=None)
      Starting value of eigenvector iteration for each node.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality(G)
    >>> sorted((v, f"{c:0.2f}") for v, c in centrality.items())
    [(0, '0.37'), (1, '0.60'), (2, '0.60'), (3, '0.37')]

    Notes
    -----
    The measure was introduced by [1]_ and is discussed in [2]_.

    The power iteration method is used to compute the eigenvector and
    convergence is **not** guaranteed. Our method stops after ``max_iter``
    iterations or when the change in the computed vector between two
    iterations is smaller than an error tolerance of
    ``G.number_of_nodes() * tol``. This implementation uses ($A + I$)
    rather than the adjacency matrix $A$ because it shifts the spectrum
    to enable discerning the correct eigenvector even for networks with
    multiple dominant eigenvalues.

    For directed graphs this is "left" eigenvector centrality which corresponds
    to the in-edges in the graph. For out-edges eigenvector centrality
    first reverse the graph with ``G.reverse()``.

    References
    ----------
    .. [1] Phillip Bonacich.
       "Power and Centrality: A Family of Measures."
       *American Journal of Sociology* 92(5):1170–1182, 1986
       <http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf>
    .. [2] Mark E. J. Newman.
       *Networks: An Introduction.*
       Oxford University Press, USA, 2010, pp. 169.

    """
    nodes = G.nodes
    if len(nodes) == 0:
        print("cannot compute centrality for the null graph")
        exit()
    # If no initial vector is provided, start with the all-ones vector.
    if nstart is None:
        nstart = {v: 1 for v in nodes}
    # Normalize the initial vector so that each entry is in [0, 1]. This is
    # guaranteed to never have a divide-by-zero error by the previous line.
    nstart_sum = sum(nstart.values())
    x = {k: v / nstart_sum for k, v in nstart.items()}
    nnodes = len(nodes)
    adj=G.adj
    # make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = xlast.copy()  # Start with xlast times I to iterate with (A+I)
        # do the multiplication y^T = x^T A (left eigenvector)
        for n in x:
            for nbr in adj[n]:
                w = G[n][nbr].get("weight", 1) if weight else 1
                x[nbr] += xlast[n] * w
        # Normalize the vector. The normalization denominator `norm`
        # should never be zero by the Perron--Frobenius
        # theorem. However, in case it is due to numerical error, we
        # assume the norm to be one instead.
        norm = sqrt(sum(z ** 2 for z in x.values())) or 1
        x = {k: v / norm for k, v in x.items()}
        # Check for convergence (in the L_1 norm).
        if sum(abs(x[n] - xlast[n]) for n in x) < nnodes * tol:
            return x
    print("alreay up to max_iter")
    exit()

# not_implemented_for('directed')
def current_flow_betweenness_centrality(G, normalized=True, weight=None,
                                        dtype=float, solver='full'):
    r"""Compute current-flow betweenness centrality for nodes.

    Current-flow betweenness centrality uses an electrical current
    model for information spreading in contrast to betweenness
    centrality which uses shortest paths.

    Current-flow betweenness centrality is also known as
    random-walk betweenness centrality [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by 2/[(n-1)(n-2)] where
      n is the number of nodes in G.

    weight : string or None, optional (default=None)
      Key for edge data used as the edge weight.
      If None, then use 1 as each edge weight.

    dtype : data type (float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver : string (default='lu')
       Type of linear solver to use for computing the flow matrix.
       Options are "full" (uses most memory), "lu" (recommended), and
       "cg" (uses least memory).

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    Notes
    -----
    Current-flow betweenness can be computed in  $O(I(n-1)+mn \log n)$
    time [1]_, where $I(n-1)$ is the time needed to compute the
    inverse Laplacian.  For a full matrix this is $O(n^3)$ but using
    sparse methods you can achieve $O(nm{\sqrt k})$ where $k$ is the
    Laplacian matrix condition number.

    The space required is $O(nw)$ where $w$ is the width of the sparse
    Laplacian matrix.  Worse case is $w=n$ for $O(n^2)$.

    If the edges have a 'weight' attribute they will be used as
    weights in this algorithm.  Unspecified weights are set to 1.

    References
    ----------
    .. [1] Centrality Measures Based on Current Flow.
       Ulrik Brandes and Daniel Fleischer,
       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS '05).
       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.
       http://algo.uni-konstanz.de/publications/bf-cmbcf-05.pdf

    .. [2] A measure of betweenness centrality based on random walks,
       M. E. J. Newman, Social Networks 27, 39-54 (2005).
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError('current_flow_betweenness_centrality requires NumPy ',
                          'http://scipy.org/')
    try:
        import scipy
    except ImportError:
        raise ImportError('current_flow_betweenness_centrality requires SciPy ',
                          'http://scipy.org/')
    if not eg.is_connected(G):
        print("Graph not connected.")
        exit()
    n = len(G)
    nodes = G.nodes
    result_dict = dict()
    for node in nodes:
        result_dict[node] = 0
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    H=eg.Graph()
    Match = dict(zip(ordering,range(n)))
    for u, v, d in G.edges:
        H.add_edge(Match[u],Match[v],weight = d.get("weight",1))
    result_dict = dict.fromkeys(H, 0.0)
    for row, (s, t) in flow_matrix_row(H, weight=weight, dtype=dtype,
                                       solver=solver):
        pos = dict(zip(row.argsort()[::-1], range(n)))
        for i in range(n):
            result_dict[s] += (i - pos[i]) * row[i]
            result_dict[t] += (n - i - 1 - pos[i]) * row[i]
    if normalized:
        nb = (n - 1.0) * (n - 2.0)  # normalization factor
    else:
        nb = 2.0
    for v in H:
        result_dict[v] = float((result_dict[v] - v) * 2.0 / nb)
    return {ordering[k]: v for k, v in result_dict.items()}

# not_implemented_for('directed')
def current_flow_closeness_centrality(G, weight=None,
                                      dtype=float, solver='lu'):
    """Compute current-flow closeness centrality for nodes.

    Current-flow closeness centrality is variant of closeness
    centrality based on effective resistance between nodes in
    a network. This metric is also known as information centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    dtype: data type (default=float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver: string (default='lu')
       Type of linear solver to use for computing the flow matrix.
       Options are "full" (uses most memory), "lu" (recommended), and
       "cg" (uses least memory).

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with current flow closeness centrality as the value.

    See Also
    --------
    closeness_centrality

    Notes
    -----
    The algorithm is from Brandes [1]_.

    See also [2]_ for the original definition of information centrality.

    References
    ----------
    .. [1] Ulrik Brandes and Daniel Fleischer,
       Centrality Measures Based on Current Flow.
       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS '05).
       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.
       http://algo.uni-konstanz.de/publications/bf-cmbcf-05.pdf

    .. [2] Karen Stephenson and Marvin Zelen:
       Rethinking centrality: Methods and examples.
       Social Networks 11(1):1-37, 1989.
       https://doi.org/10.1016/0378-8733(89)90016-6
    """
    import numpy as np
    import scipy
    if not eg.is_connected(G):
        raise nx.NetworkXError("Graph not connected.")
    solvername = {"full": FullInverseLaplacian,
                  "lu": SuperLUInverseLaplacian,
                  "cg": CGInverseLaplacian}
    n = len(G)
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    H=eg.Graph()
    Match = dict(zip(ordering,range(n)))
    for u, v, d in G.edges:
        H.add_edge(Match[u],Match[v],weight = d.get("weight",1))
    result_dict = dict.fromkeys(H, 0.0)
    L = laplacian_sparse_matrix(H, nodelist=range(n), weight=weight,
                                dtype=dtype, format='csc')
    C2 = solvername[solver](L, width=1, dtype=dtype)  # initialize solver
    for v in H:
        col = C2.get_row(v)
        for w in H:
            result_dict[v] += col[v] - 2 * col[w]
            result_dict[w] += col[v]
    for v in H:
        result_dict[v] = 1.0 / (result_dict[v])
    return {ordering[k]: float(v) for k, v in result_dict.items()}

#not_implemented_for('directed')
def second_order_centrality(G):
      
  
    """Compute the second order centrality for nodes of G.

    The second order centrality of a given node is the standard deviation of
    the return times to that node of a perpetual random walk on G:

    Parameters
    ----------
    G : graph
      A NetworkX connected and undirected graph.

    Returns
    -------
    nodes : dictionary
       Dictionary keyed by node with second order centrality as the value.

    Examples
    --------
    >>> G = nx.star_graph(10)
    >>> soc = nx.second_order_centrality(G)
    >>> print(sorted(soc.items(), key=lambda x:x[1])[0][0]) # pick first id
    0

    Raises
    ------
    NetworkXException
        If the graph G is empty, non connected or has negative weights.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    Lower values of second order centrality indicate higher centrality.

    The algorithm is from Kermarrec, Le Merrer, Sericola and Trédan [1]_.

    This code implements the analytical version of the algorithm, i.e.,
    there is no simulation of a random walk process involved. The random walk
    is here unbiased (corresponding to eq 6 of the paper [1]_), thus the
    centrality values are the standard deviations for random walk return times
    on the transformed input graph G (equal in-degree at each nodes by adding
    self-loops).

    Complexity of this implementation, made to run locally on a single machine,
    is O(n^3), with n the size of G, which makes it viable only for small
    graphs.

    References
    ----------
    .. [1] Anne-Marie Kermarrec, Erwan Le Merrer, Bruno Sericola, Gilles Trédan
       "Second order centrality: Distributed assessment of nodes criticity in
       complex networks", Elsevier Computer Communications 34(5):619-628, 2011.
    """
    if G.is_directed() == True:
        print("not implemented for directed type!" )
        exit()
    try:
        import numpy as np
    except ImportError:
        raise ImportError('Requires NumPy: http://scipy.org/')

    n = len(G)

    if n == 0:
        print("Empty graph.")
        exit()
    if not eg.is_connected(G):
        print("Non connected graph.")
        exit
    if any(d.get('weight', 0) < 0 for u, v, d in G.edges):
        print("Graph has negative edge weights.")
        exit()
    H = eg.DiGraph()
    # balancing G for Metropolis-Hastings random walks
    for u, v, d in G.edges:
        H.add_edge(u, v, weight = d.get("weight",1))  
        H.add_edge(v, u, weight = d.get("weight",1))
    deg = dict(H.in_degree(weight='weight'))
    d_max = max(deg.values())
    for i, deg in deg.items():
        if deg < d_max:
            H.add_edge(i, i, weight=d_max-deg)
    P = to_numpy_matrix(H)
    P = P / P.sum(axis=1)  # to transition probability matrix

    def _Qj(P, j):
        P = P.copy()
        P[:, j] = 0
        return P

    M = np.empty([n, n])

    for i in range(n):
        M[:, i] = np.linalg.solve(np.identity(n) - _Qj(P, i),
                                  np.ones([n, 1])[:, 0])  # eq 3

    return dict(zip(G.nodes,
                    [np.sqrt(2*np.sum(M[:, i])-n*(n+1)) for i in range(n)]
                    ))  

# not_implemented_for('directed')
# not_implemented_for('multigraph')
def communicability_betweenness_centrality(G, normalized=True):
    """Returns subgraph communicability for all pairs of nodes in G.

    Communicability betweenness measure makes use of the number of walks
    connecting every pair of nodes as the basis of a betweenness centrality
    measure.

    Parameters
    ----------
    G: graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with communicability betweenness as the value.

    Notes
    -----
    Let `G=(V,E)` be a simple undirected graph with `n` nodes and `m` edges,
    and `A` denote the adjacency matrix of `G`.

    Let `G(r)=(V,E(r))` be the graph resulting from
    removing all edges connected to node `r` but not the node itself.

    The adjacency matrix for `G(r)` is `A+E(r)`,  where `E(r)` has nonzeros
    only in row and column `r`.

    The subraph betweenness of a node `r`  is [1]_

    .. math::

         \omega_{r} = \frac{1}{C}\sum_{p}\sum_{q}\frac{G_{prq}}{G_{pq}},
         p\neq q, q\neq r,

    where
    `G_{prq}=(e^{A}_{pq} - (e^{A+E(r)})_{pq}`  is the number of walks
    involving node r,
    `G_{pq}=(e^{A})_{pq}` is the number of closed walks starting
    at node `p` and ending at node `q`,
    and `C=(n-1)^{2}-(n-1)` is a normalization factor equal to the
    number of terms in the sum.

    The resulting `\omega_{r}` takes values between zero and one.
    The lower bound cannot be attained for a connected
    graph, and the upper bound is attained in the star graph.

    References
    ----------
    .. [1] Ernesto Estrada, Desmond J. Higham, Naomichi Hatano,
       "Communicability Betweenness in Complex Networks"
       Physica A 388 (2009) 764-774.
       https://arxiv.org/abs/0905.4102

    """
    import numpy as np
    import scipy.linalg

    nodelist = list(G)  # ordering of nodes in matrix
    n = len(nodelist)
    A = to_numpy_matrix(G, nodelist)
    # convert to 0-1 matrix
    A[A != 0.0] = 1
    expA = scipy.linalg.expm(A.A)
    mapping = dict(zip(nodelist, range(n)))
    cbc = {}
    for v in G:
        # remove row and col of node v
        i = mapping[v]
        row = A[i, :].copy()
        col = A[:, i].copy()
        A[i, :] = 0
        A[:, i] = 0
        B = (expA - scipy.linalg.expm(A.A)) / expA
        # sum with row/col of node v and diag set to zero
        B[i, :] = 0
        B[:, i] = 0
        B -= np.diag(np.diag(B))
        cbc[v] = float(B.sum())
        # put row and col back
        A[i, :] = row
        A[:, i] = col
    # rescaling
    cbc = _rescale(cbc, normalized=normalized)
    return cbc

def _rescale(cbc, normalized):
    # helper to rescale betweenness centrality
    if normalized is True:
        order = len(cbc)
        if order <= 2:
            scale = None
        else:
            scale = 1.0 / ((order - 1.0) ** 2 - (order - 1.0))
    if scale is not None:
        for v in cbc:
            cbc[v] *= scale
    return cbc

def group_betweenness_centrality(G, C, normalized=True, weight=None):
    r"""Compute the group betweenness centrality for a group of nodes.

    Group betweenness centrality of a group of nodes $C$ is the sum of the
    fraction of all-pairs shortest paths that pass through any vertex in $C$

    .. math::

       c_B(C) =\sum_{s,t \in V-C; s<t} \frac{\sigma(s, t|C)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\sigma(s, t|C)$ is the number of
    those paths passing through some node in group $C$. Note that
    $(s, t)$ are not members of the group ($V-C$ is the set of nodes
    in $V$ that are not in $C$).

    Parameters
    ----------
    G : graph
      A easygraph graph.

    C : list or set
      C is a group of nodes which belong to G, for which group betweenness
      centrality is to be calculated.

    normalized : bool, optional
      If True, group betweenness is normalized by `2/((|V|-|C|)(|V|-|C|-1))`
      for graphs and `1/((|V|-|C|)(|V|-|C|-1))` for directed graphs where `|V|`
      is the number of nodes in G and `|C|` is the number of nodes in C.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Raises
    ------
    NodeNotFound
       If node(s) in C are not present in G.

    Returns
    -------
    betweenness : float
       Group betweenness centrality of the group C.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    The measure is described in [1]_.
    The algorithm is an extension of the one proposed by Ulrik Brandes for
    betweenness centrality of nodes. Group betweenness is also mentioned in
    his paper [2]_ along with the algorithm. The importance of the measure is
    discussed in [3]_.

    The number of nodes in the group must be a maximum of n - 2 where `n`
    is the total number of nodes in the graph.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] Ulrik Brandes:
       On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.9610&rep=rep1&type=pdf
    .. [3] Sourav Medya et. al.:
       Group Centrality Maximization via Network Design.
       SIAM International Conference on Data Mining, SDM 2018, 126–134.
       https://sites.cs.ucsb.edu/~arlei/pubs/sdm18.pdf
    """
    
    betweenness = 0  # initialize betweenness to 0
    V = set(G)  # set of nodes in G
    C = set(C)  # set of nodes in C (group)
    if len(C - V) != 0:  # element(s) of C not in V
        print('The node(s) ' + str(list(C - V)) + ' are not '
                              'in the graph.')
        exit()
    V_C = V - C  # set of nodes in V but not in C
    # accumulation
    for pair in combinations(V_C, 2):  # (s, t) pairs of V_C
        paths = 0
        paths_through_C = 0
        for path in all_shortest_paths(G, source=pair[0], target=pair[1], weight=weight):
            if set(path) & C:
                paths_through_C += 1
            paths += 1
        if paths != 0:
            betweenness += paths_through_C / paths
        else:
            betweenness += 0
    # rescaling
    v, c = len(G), len(C)
    if normalized:
        scale = 1 / ((v - c) * (v - c - 1))
        if not G.is_directed():
            scale *= 2
    else:
        scale = None
    if scale is not None:
        betweenness *= scale
    return betweenness

# not_implemented_for('directed')
def group_closeness_centrality(G, S):
    r"""Compute the group closeness centrality for a group of nodes.

    Group closeness centrality of a group of nodes $S$ is a measure
    of how close the group is to the other nodes in the graph.

    .. math::

       c_{close}(S) = \frac{|V-S|}{\sum_{v \in V-S} d_{S, v}}

       d_{S, v} = min_{u \in S} (d_{u, v})

    where $V$ is the set of nodes, $d_{S, v}$ is the distance of
    the group $S$ from $v$ defined as above. ($V-S$ is the set of nodes
    in $V$ that are not in $S$).

    Parameters
    ----------
    G : graph
       A easygraph graph.

    S : list or set
       S is a group of nodes which belong to G, for which group closeness
       centrality is to be calculated.

    weight : None or string, optional (default=None)
       If None, all edge weights are considered equal.
       Otherwise holds the name of the edge attribute used as weight.

    Raises
    ------
    NodeNotFound
       If node(s) in S are not present in G.

    Returns
    -------
    closeness : float
       Group closeness centrality of the group S.

    See Also
    --------
    closeness_centrality

    Notes
    -----
    The measure was introduced in [1]_.
    The formula implemented here is described in [2]_.

    Higher values of closeness indicate greater centrality.

    It is assumed that 1 / 0 is 0 (required in the case of directed graphs,
    or when a shortest path length is 0).

    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    For directed graphs, the incoming distance is utilized here. To use the
    outward distance, act on `G.reverse()`.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] J. Zhao et. al.:
       Measuring and Maximizing Group Closeness Centrality over
       Disk Resident Graphs.
       WWWConference Proceedings, 2014. 689-694.
       http://wwwconference.org/proceedings/www2014/companion/p689.pdf
    """
    if G.is_directed():
        print(" not_implemented_for directed ")
        exit()
    closeness = 0  # initialize to 0
    V = set(G)  # set of nodes in G
    S = set(S)  # set of nodes in group S
    V_S = V - S  # set of nodes in V but not S
    _, _, shortest_path_lengths = Dijkstra_mutilsource_path_predecessor_distance(G, S)
    # accumulation
    for v in V_S:
        try:
            closeness += shortest_path_lengths[v]
        except KeyError:  # no path exists
            closeness += 0
    try:
        closeness = len(V_S) / closeness
    except ZeroDivisionError:  # 1 / 0 assumed as 0
        closeness = 0
    return closeness

def group_degree_centrality(G, S):
    """Compute the group degree centrality for a group of nodes.

    Group degree centrality of a group of nodes $S$ is the fraction
    of non-group members connected to group members.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group degree
       centrality is to be calculated.

    Raises
    ------
    NetworkXError
       If node(s) in S are not in G.

    Returns
    -------
    centrality : float
       Group degree centrality of the group S.

    See Also
    --------
    degree_centrality
    group_in_degree_centrality
    group_out_degree_centrality

    Notes
    -----
    The measure was introduced in [1]_.

    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    """
    centrality = len(set().union(*(set(G.adj[i])
                                       for i in S)) - set(S))
    centrality /= (len(G) - len(S))
    return centrality

def newman_betweenness_centrality(G, v=None, normalized=True):
    """Compute load centrality for nodes.

    The load centrality of a node is the fraction of all shortest
    paths that pass through that node.

    Parameters
    ----------
    G : graph
      A easygraph graph.

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by b=b/(n-1)(n-2) where
      n is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, edge weights are ignored.
      Otherwise holds the name of the edge attribute used as weight.

    cutoff : bool, optional (default=None)
      If specified, only consider paths of length <= cutoff.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with centrality as the value.

    See Also
    --------
    betweenness_centrality()

    Notes
    -----
    Load centrality is slightly different than betweenness. It was originally
    introduced by [2]_. For this load algorithm see [1]_.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Scientific collaboration networks. II.
       Shortest paths, weighted networks, and centrality.
       Physical Review E 64, 016132, 2001.
       http://journals.aps.org/pre/abstract/10.1103/PhysRevE.64.016132
    .. [2] Kwang-Il Goh, Byungnam Kahng and Doochul Kim
       Universal behavior of Load Distribution in Scale-Free Networks.
       Physical Review Letters 87(27):1–4, 2001.
       http://phya.snu.ac.kr/~dkim/PRL87278701.pdf
    """
    if v is not None:   # only one node
        betweenness = 0.0
        for source in G:
            ubetween = _node_betweenness(G, source, False)
            betweenness += ubetween[v] if v in ubetween else 0
    else:
        betweenness = {}.fromkeys(G, 0.0)
        for source in betweenness:
            ubetween = _node_betweenness(G, source, False)
            for vk in ubetween:
                betweenness[vk] += ubetween[vk]
    if normalized:
        n = len(G)
        if n <= 2:
            return betweenness  # no normalization b=0 for all nodes
        scale = 1.0 / ((n - 1) * (n - 2))
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness  # all nodes

def _node_betweenness(G, source, normalized=True):
    """Node betweenness_centrality helper:

    See betweenness_centrality for what you probably want.
    This actually computes "load" and not betweenness.
    See https://networkx.lanl.gov/ticket/103

    This calculates the load of each node for paths from a single source.
    (The fraction of number of shortests paths from source that go
    through each node.)

    To get the load for a node you need to do all-pairs shortest paths.

    If weight is not None then use Dijkstra for finding shortest paths.
    """
    # get the predecessor and path length data
    (_, pred, length) = Dijkstra_mutilsource_predecessor_and_distance(G, [source])
    # order the nodes by path length
    onodes = [(l, vert) for (vert, l) in length.items()]
    onodes.sort()
    onodes[:] = [vert for (l, vert) in onodes if l > 0]

    # initialize betweenness
    between = {}.fromkeys(length, 1.0)

    while onodes:
        v = onodes.pop()
        if v in pred:
            num_paths = len(pred[v])  # Discount betweenness if more than
            for x in pred[v]:         # one shortest path.
                if x == source:  # stop if hit source because all remaining v
                    break        # also have pred[v]==[source]
                between[x] += between[v] / float(num_paths)
    #  remove source
    for v in between:
        between[v] -= 1
    # rescale to be between 0 and 1
    if normalized:
        l = len(between)
        if l > 2:
            # scale by 1/the number of possible paths
            scale = 1.0 / float((l - 1) * (l - 2))
            for v in between:
                between[v] *= scale
    return between

load_centrality = newman_betweenness_centrality

# not_implemented_for("directed")
# not_implemented_for("multigraph")
def subgraph_centrality(G):
    r"""Returns subgraph centrality for each node in G.

    Subgraph centrality  of a node `n` is the sum of weighted closed
    walks of all lengths starting and ending at node `n`. The weights
    decrease with path length. Each closed walk is associated with a
    connected subgraph ([1]_).

    Parameters
    ----------
    G: graph

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with subgraph centrality as the value.

    Raises
    ------
    NetworkXError
       If the graph is not undirected and simple.

    See Also
    --------
    subgraph_centrality_exp:
        Alternative algorithm of the subgraph centrality for each node of G.

    Notes
    -----
    This version of the algorithm computes eigenvalues and eigenvectors
    of the adjacency matrix.

    Subgraph centrality of a node `u` in G can be found using
    a spectral decomposition of the adjacency matrix [1]_,

    .. math::

       SC(u)=\sum_{j=1}^{N}(v_{j}^{u})^2 e^{\lambda_{j}},

    where `v_j` is an eigenvector of the adjacency matrix `A` of G
    corresponding corresponding to the eigenvalue `\lambda_j`.

    Examples
    --------
    (Example from [1]_)
    >>> G = nx.Graph(
    ...     [
    ...         (1, 2),
    ...         (1, 5),
    ...         (1, 8),
    ...         (2, 3),
    ...         (2, 8),
    ...         (3, 4),
    ...         (3, 6),
    ...         (4, 5),
    ...         (4, 7),
    ...         (5, 6),
    ...         (6, 7),
    ...         (7, 8),
    ...     ]
    ... )
    >>> sc = nx.subgraph_centrality(G)
    >>> print([f"{node} {sc[node]:0.2f}" for node in sorted(sc)])
    ['1 3.90', '2 3.90', '3 3.64', '4 3.71', '5 3.64', '6 3.71', '7 3.64', '8 3.90']

    References
    ----------
    .. [1] Ernesto Estrada, Juan A. Rodriguez-Velazquez,
       "Subgraph centrality in complex networks",
       Physical Review E 71, 056103 (2005).
       https://arxiv.org/abs/cond-mat/0504730

    """
    import numpy as np
    import numpy.linalg

    nodelist = list(G)  # ordering of nodes in matrix
    A = to_numpy_matrix(G, nodelist)
    # convert to 0-1 matrix
    A[A != 0.0] = 1
    w, v = numpy.linalg.eigh(A.A)
    vsquare = np.array(v) ** 2
    expw = np.exp(w)
    xg = np.dot(vsquare, expw)
    # convert vector dictionary keyed by node
    sc = dict(zip(nodelist, map(float, xg)))
    return sc

# not_implemented_for('directed')
def harmonic_centrality(G, nbunch=None, distance=None):
    r"""Compute harmonic centrality for nodes.

    Harmonic centrality [1]_ of a node `u` is the sum of the reciprocal
    of the shortest path distances from all other nodes to `u`

    .. math::

        C(u) = \sum_{v \neq u} \frac{1}{d(v, u)}

    where `d(v, u)` is the shortest-path distance between `v` and `u`.

    Notice that higher values indicate higher centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    nbunch : container
      Container of nodes. If provided harmonic centrality will be computed
      only over the nodes in nbunch.

    distance : edge attribute key, optional (default=None)
      Use the specified edge attribute as the edge distance in shortest
      path calculations.  If `None`, then each edge will have distance equal to 1.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with harmonic centrality as the value.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality, closeness_centrality

    Notes
    -----
    If the 'distance' keyword is set to an edge attribute key then the
    shortest-path length will be computed using Dijkstra's algorithm with
    that edge attribute as the edge weight.

    References
    ----------
    .. [1] Boldi, Paolo, and Sebastiano Vigna. "Axioms for centrality."
           Internet Mathematics 10.3-4 (2014): 222-262.
    """
    if G.is_directed():
        print("not implemented for directed ")
        exit()
    dist = GetAllDistance(G)
    return {
        u: sum(1 / d if d > 0 else 0 for v, d in dist[u].items())
        for u in G.nodes
    }

def dispersion(G, u=None, v=None, normalized=True, alpha=1.0, b=0.0, c=0.0):
    r"""Calculate dispersion between `u` and `v` in `G`.

    A link between two actors (`u` and `v`) has a high dispersion when their
    mutual ties (`s` and `t`) are not well connected with each other.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    u : node, optional
        The source for the dispersion score (e.g. ego node of the network).
    v : node, optional
        The target of the dispersion score if specified.
    normalized : bool
        If True (default) normalize by the embededness of the nodes (u and v).

    Returns
    -------
    nodes : dictionary
        If u (v) is specified, returns a dictionary of nodes with dispersion
        score for all "target" ("source") nodes. If neither u nor v is
        specified, returns a dictionary of dictionaries for all nodes 'u' in the
        graph with a dispersion score for each node 'v'.

    Notes
    -----
    This implementation follows Lars Backstrom and Jon Kleinberg [1]_. Typical
    usage would be to run dispersion on the ego network $G_u$ if $u$ were
    specified.  Running :func:`dispersion` with neither $u$ nor $v$ specified
    can take some time to complete.

    References
    ----------
    .. [1] Romantic Partnerships and the Dispersion of Social Ties:
        A Network Analysis of Relationship Status on Facebook.
        Lars Backstrom, Jon Kleinberg.
        https://arxiv.org/pdf/1310.6753v1.pdf

    """

    def _dispersion(G_u, u, v):
        """dispersion for all nodes 'v' in a ego network G_u of node 'u'"""
        u_nbrs = set(G_u[u])
        ST = {n for n in G_u[v] if n in u_nbrs}
        set_uv = {u, v}
        # all possible ties of connections that u and b share
        possib = combinations(ST, 2)
        total = 0
        for (s, t) in possib:
            # neighbors of s that are in G_u, not including u and v
            nbrs_s = u_nbrs.intersection(G_u[s]) - set_uv
            # s and t are not directly connected
            if t not in nbrs_s:
                # s and t do not share a connection
                if nbrs_s.isdisjoint(G_u[t]):
                    # tick for disp(u, v)
                    total += 1
        # neighbors that u and v share
        embededness = len(ST)

        if normalized:
            if embededness + c != 0:
                norm_disp = ((total + b) ** alpha) / (embededness + c)
            else:
                norm_disp = (total + b) ** alpha
            dispersion = norm_disp

        else:
            dispersion = total

        return dispersion

    if u is None:
        # v and u are not specified
        if v is None:
            results = {n: {} for n in G}
            for u in G:
                for v in G.adj[u]:
                    results[u][v] = _dispersion(G.adj, u, v)
        # u is not specified, but v is
        else:
            results = dict.fromkeys(G[v], {})
            for u in G.adj[v]:
                results[u] = _dispersion(G.adj, v, u)
    else:
        # u is specified with no target v
        if v is None:
            results = dict.fromkeys(G[u], {})
            for v in G.adj[u]:
                results[v] = _dispersion(G.adj, u, v)
        # both u and v are specified
        else:
            results = _dispersion(G.adj, u, v)

    return results

def global_reaching_centrality(G, normalized=True, weight=None):
    """Returns the global reaching centrality of a directed graph.

    The *global reaching centrality* of a weighted directed graph is the
    average over all nodes of the difference between the local reaching
    centrality of the node and the greatest local reaching centrality of
    any node in the graph [1]_. For more information on the local
    reaching centrality, see :func:`local_reaching_centrality`.
    Informally, the local reaching centrality is the proportion of the
    graph that is reachable from the neighbors of the node.

    Parameters
    ----------
    G : DiGraph
        A networkx DiGraph.

    weight : None or string, optional (default=None)
        Attribute to use for edge weights. If ``None``, each edge weight
        is assumed to be one. A higher weight implies a stronger
        connection between nodes and a *shorter* path length.

    normalized : bool, optional (default=True)
        Whether to normalize the edge weights by the total sum of edge
        weights.

    Returns
    -------
    h : float
        The global reaching centrality of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2)
    >>> G.add_edge(1, 3)
    >>> nx.global_reaching_centrality(G)
    1.0
    >>> G.add_edge(3, 2)
    >>> nx.global_reaching_centrality(G)
    0.75

    See also
    --------
    local_reaching_centrality

    References
    ----------
    .. [1] Mones, Enys, Lilla Vicsek, and Tamás Vicsek.
           "Hierarchy Measure for Complex Networks."
           *PLoS ONE* 7.3 (2012): e33799.
           https://doi.org/10.1371/journal.pone.0033799
    """
    if any(d.get("weight",1) for u, v, d in G.edges) < 0:
          print("edge weights must be positive")
          exit()
    total_weight = G.size(weight=weight)
    if weight is not None:
        def as_distance(u, v, d): return total_weight / d.get("weight", 1)
        shortest_paths = dict( all_pairs_dijkstra_path(G, weight=as_distance))
    else:
        shortest_paths = dict( all_pairs_dijkstra_path(G))
    centrality = local_reaching_centrality
    # TODO This can be trivially parallelized.
    lrc = [centrality(G, node, paths=paths, weight=weight,
                      normalized=normalized)
           for node, paths in shortest_paths.items()]
    max_lrc = max(lrc)
    return sum(max_lrc - c for c in lrc) / (len(G) - 1)

def local_reaching_centrality(G, v, weight=None, paths=None, normalized=True):
    """Returns the local reaching centrality of a node in a directed
    graph.

    The *local reaching centrality* of a node in a directed graph is the
    proportion of other nodes reachable from that node [1]_.

    Parameters
    ----------
    G : DiGraph
        A NetworkX DiGraph.

    v : node
        A node in the directed graph `G`.

    paths : dictionary (default=None)
        If this is not `None` it must be a dictionary representation
        of single-source shortest paths, as computed by, for example,
        :func:`networkx.shortest_path` with source node `v`. Use this
        keyword argument if you intend to invoke this function many
        times but don't want the paths to be recomputed each time.

    weight : None or string, optional (default=None)
        Attribute to use for edge weights.  If `None`, each edge weight
        is assumed to be one. A higher weight implies a stronger
        connection between nodes and a *shorter* path length.

    normalized : bool, optional (default=True)
        Whether to normalize the edge weights by the total sum of edge
        weights.

    Returns
    -------
    h : float
        The local reaching centrality of the node ``v`` in the graph
        ``G``.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3)])
    >>> nx.local_reaching_centrality(G, 3)
    0.0
    >>> G.add_edge(3, 2)
    >>> nx.local_reaching_centrality(G, 3)
    0.5

    See also
    --------
    global_reaching_centrality

    References
    ----------
    .. [1] Mones, Enys, Lilla Vicsek, and Tamás Vicsek.
           "Hierarchy Measure for Complex Networks."
           *PLoS ONE* 7.3 (2012): e33799.
           https://doi.org/10.1371/journal.pone.0033799
    """  
    if paths is None:
        if any(d.get("weight",1) for u, v, d in G.edges) < 0:
            print("edge weights must be positive")
            exit()
        total_weight = G.size(weight=weight)
        if weight is not None:
            def as_distance(u, v, d): return total_weight / d.get("weight", 1)
            paths = dict( all_pairs_dijkstra_path(G, weight=as_distance))
        else:
            paths = dict( all_pairs_dijkstra_path(G))
    # If the graph is unweighted, simply return the proportion of nodes
    # reachable from the source node ``v``.
    if weight is None and G.is_directed():
        return (len(paths) - 1) / (len(G) - 1)
    if normalized and weight is not None:
        norm = G.size(weight=weight) / G.size()
    else:
        norm = 1
    # TODO This can be trivially parallelized.
    avgw = (_average_weight(G, path, weight=weight) for path in paths.values())
    sum_avg_weight = sum(avgw) / norm
    return sum_avg_weight / (len(G) - 1)

def _average_weight(G, path, weight=None):
    """Returns the average weight of an edge in a weighted path.

    Parameters
    ----------
    G : graph
      A networkx graph.

    path: list
      A list of vertices that define the path.

    weight : None or string, optional (default=None)
      If None, edge weights are ignored.  Then the average weight of an edge
      is assumed to be the multiplicative inverse of the length of the path.
      Otherwise holds the name of the edge attribute used as weight.
    """
    path_length = len(path) - 1
    if path_length <= 0:
        return 0
    if weight is None:
        return 1 / path_length
    total_weight = sum(G.adj[i][j].get("weight",1) for i, j in pairwise(path))
    return total_weight / path_length

# Recipe from the itertools documentation.
def pairwise(iterable, cyclic=False):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    first = next(b, None)
    if cyclic is True:
        return zip(a, chain(b, (first,)))
    return zip(a, b)