import easygraph as eg
import itertools

__all__ = [
    "from_numpy_array",
    "to_numpy_matrix",
    "to_numpy_array",
]


def to_numpy_matrix(G, edge_sign=1.0, not_edge_sign=0.0):
    """
    Returns the graph adjacency matrix as a NumPy matrix.

    Parameters
    ----------
    edge_sign : float
        Sign for the position of matrix where there is an edge
    
    not_edge_sign : float
        Sign for the position of matrix where there is no edge

    """
    import numpy as np
    index_of_node = dict(zip(G.nodes, range(len(G))))
    N = len(G)
    M = np.full((N, N), not_edge_sign)

    for u, udict in G.adj.items():
        for v, data in udict.items():
            M[index_of_node[u], index_of_node[v]] = edge_sign

    M = np.asmatrix(M)
    return M


def from_numpy_array(A, parallel_edges=False, create_using=None):
    """Returns a graph from a 2D NumPy array.

    The 2D NumPy array is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    A : a 2D numpy.ndarray
        An adjacency matrix representation of a graph

    parallel_edges : Boolean
        If this is True, `create_using` is a multigraph, and `A` is an
        integer array, then entry *(i, j)* in the array is interpreted as the
        number of parallel edges joining vertices *i* and *j* in the graph.
        If it is False, then the entries in the array are interpreted as
        the weight of a single edge joining the vertices.

    create_using : EasyGraph graph constructor, optional (default=eg.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    For directed graphs, explicitly mention create_using=eg.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`easygraph.MultiGraph` or
    :class:`easygraph.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (of the same type as `create_using`) with parallel edges.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the array `A` will be added to the
    graph.

    If the NumPy array has a single data type for each array entry it
    will be converted to an appropriate Python data type.

    If the NumPy array has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    EasyGraph graph.

    See Also
    --------
    to_numpy_array

    Examples
    --------
    Simple integer weights on edges:

    >>> import numpy as np
    >>> A = np.array([[1, 1], [2, 1]])
    >>> G = eg.from_numpy_array(A)
    >>> G.edges(data=True)
    EdgeDataView([(0, 0, {'weight': 1}), (0, 1, {'weight': 2}), (1, 1, {'weight': 1})])

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = np.array([[1, 1], [1, 2]])
    >>> G = eg.from_numpy_array(A, create_using=eg.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = np.array([[1, 1], [1, 2]])
    >>> temp = eg.MultiGraph()
    >>> G = eg.from_numpy_array(A, parallel_edges=True, create_using=temp)
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    User defined compound data type on edges:

    >>> dt = [("weight", float), ("cost", int)]
    >>> A = np.array([[(1.0, 2)]], dtype=dt)
    >>> G = eg.from_numpy_array(A)
    >>> G.edges()
    EdgeView([(0, 0)])
    >>> G[0][0]["cost"]
    2
    >>> G[0][0]["weight"]
    1.0

    """
    kind_to_python_type = {
        "f": float,
        "i": int,
        "u": int,
        "b": bool,
        "c": complex,
        "S": str,
        "U": str,
        "V": "void",
    }
    G = eg.empty_graph(0, create_using)
    if A.ndim != 2:
        raise eg.EasyGraphError(f"Input array must be 2D, not {A.ndim}")
    n, m = A.shape
    if n != m:
        raise eg.EasyGraphError(
            f"Adjacency matrix not square: eg,ny={A.shape}")
    dt = A.dtype
    try:
        python_type = kind_to_python_type[dt.kind]
    except Exception as err:
        raise TypeError(f"Unknown numpy data type: {dt}") from err

    # Make sure we get even the isolated nodes of the graph.
    G.add_nodes_from(range(n))
    # Get a list of all the entries in the array with nonzero entries. These
    # coordinates become edges in the graph. (convert to int from np.int64)
    edges = ((int(e[0]), int(e[1])) for e in zip(*A.nonzero()))
    # handle numpy constructed data type
    if python_type == "void":
        # Sort the fields by their offset, then by dtype, then by name.
        fields = sorted((offset, dtype, name)
                        for name, (dtype, offset) in A.dtype.fields.items())
        triples = ((
            u,
            v,
            {
                name: kind_to_python_type[dtype.kind](val)
                for (_, dtype, name), val in zip(fields, A[u, v])
            },
        ) for u, v in edges)
    # If the entries in the adjacency matrix are integers, the graph is a
    # multigraph, and parallel_edges is True, then create parallel edges, each
    # with weight 1, for each entry in the adjacency matrix. Otherwise, create
    # one edge for each positive entry in the adjacency matrix and set the
    # weight of that edge to be the entry in the matrix.
    elif python_type is int and G.is_multigraph() and parallel_edges:
        chain = itertools.chain.from_iterable
        # The following line is equivalent to:
        #
        #     for (u, v) in edges:
        #         for d in range(A[u, v]):
        #             G.add_edge(u, v, weight=1)
        #
        triples = chain(((u, v, {
            "weight": 1
        }) for d in range(A[u, v])) for (u, v) in edges)
    else:  # basic data type
        triples = ((u, v, dict(weight=python_type(A[u, v]))) for u, v in edges)
    # If we are creating an undirected multigraph, only add the edges from the
    # upper triangle of the matrix. Otherwise, add all the edges. This relies
    # on the fact that the vertices created in the
    # `_generated_weighted_edges()` function are actually the row/column
    # indices for the matrix `A`.
    #
    # Without this check, we run into a problem where each edge is added twice
    # when `G.add_edges_from()` is invoked below.
    if G.is_multigraph() and not G.is_directed():
        triples = ((u, v, d) for u, v, d in triples if u <= v)
    G.add_edges_from(triples)
    return G


def to_numpy_array(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
):
    """Returns the graph adjacency matrix as a NumPy array.

    Parameters
    ----------
    G : graph
        The EasyGraph graph used to construct the NumPy array.

    nodelist : list, optional
        The rows and columns are ordered according to the nodes in `nodelist`.
        If `nodelist` is None, then the ordering is produced by G.nodes().

    dtype : NumPy data type, optional
        A valid single NumPy data type used to initialize the array.
        This must be a simple type such as int or numpy.float64 and
        not a compound data type (see to_numpy_recarray)
        If None, then the NumPy default is used.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. If None, then the NumPy default
        is used.

    multigraph_weight : {sum, min, max}, optional
        An operator that determines how weights in multigraphs are handled.
        The default is to sum the weights of the multiple edges.

    weight : string or None optional (default = 'weight')
        The edge attribute that holds the numerical value used for
        the edge weight. If an edge does not have that attribute, then the
        value 1 is used instead.

    nonedge : float (default = 0.0)
        The array values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are array values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as nan.

    Returns
    -------
    A : NumPy ndarray
        Graph adjacency matrix

    See Also
    --------
    from_numpy_array

    Notes
    -----
    For directed graphs, entry i,j corresponds to an edge from i to j.

    Entries in the adjacency matrix are assigned to the weight edge attribute.
    When an edge does not have a weight attribute, the value of the entry is
    set to the number 1.  For multiple (parallel) edges, the values of the
    entries are determined by the `multigraph_weight` parameter. The default is
    to sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the adjacency matrix is
    built from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal array entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute). If the
    alternate convention of doubling the edge weight is desired the
    resulting NumPy array can be modified as follows:

    >>> import numpy as np
    >>> G = eg.Graph([(1, 1)])
    >>> A = eg.to_numpy_array(G)
    >>> A
    array([[1.]])
    >>> A[np.diag_indices_from(A)] *= 2
    >>> A
    array([[2.]])

    Examples
    --------
    >>> G = eg.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> eg.to_numpy_array(G, nodelist=[0, 1, 2])
    array([[0., 2., 0.],
           [1., 0., 0.],
           [0., 0., 4.]])

    """
    import numpy as np

    if nodelist is None:
        nodelist = list(G)
        nodeset = G
        nlen = len(G)
    else:
        nlen = len(nodelist)
        nodeset = set(G.nodes)
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise eg.EasyGraphError(
                        f"Node {n} in nodelist is not in G")
            raise eg.EasyGraphError("nodelist contains duplicates.")

    undirected = not G.is_directed()
    index = dict(zip(nodelist, range(nlen)))

    # Initially, we start with an array of nans.  Then we populate the array
    # using data from the graph.  Afterwards, any leftover nans will be
    # converted to the value of `nonedge`.  Note, we use nans initially,
    # instead of zero, for two reasons:
    #
    #   1) It can be important to distinguish a real edge with the value 0
    #      from a nonedge with the value 0.
    #
    #   2) When working with multi(di)graphs, we must combine the values of all
    #      edges between any two nodes in some manner.  This often takes the
    #      form of a sum, min, or max.  Using the value 0 for a nonedge would
    #      have undesirable effects with min and max, but using nanmin and
    #      nanmax with initially nan values is not problematic at all.
    #
    # That said, there are still some drawbacks to this approach. Namely, if
    # a real edge is nan, then that value is a) not distinguishable from
    # nonedges and b) is ignored by the default combinator (nansum, nanmin,
    # nanmax) functions used for multi(di)graphs. If this becomes an issue,
    # an alternative approach is to use masked arrays.  Initially, every
    # element is masked and set to some `initial` value. As we populate the
    # graph, elements are unmasked (automatically) when we combine the initial
    # value with the values given by real edges.  At the end, we convert all
    # masked values to `nonedge`. Using masked arrays fully addresses reason 1,
    # but for reason 2, we would still have the issue with min and max if the
    # initial values were 0.0.  Note: an initial value of +inf is appropriate
    # for min, while an initial value of -inf is appropriate for max. When
    # working with sum, an initial value of zero is appropriate. Ideally then,
    # we'd want to allow users to specify both a value for nonedges and also
    # an initial value.  For multi(di)graphs, the choice of the initial value
    # will, in general, depend on the combinator function---sensible defaults
    # can be provided.

    if G.is_multigraph():
        # Handle MultiGraphs and MultiDiGraphs
        A = np.full((nlen, nlen), np.nan, order=order)
        # use numpy nan-aware operations
        operator = {sum: np.nansum, min: np.nanmin, max: np.nanmax}
        try:
            op = operator[multigraph_weight]
        except Exception as err:
            raise ValueError(
                "multigraph_weight must be sum, min, or max") from err

        for u, v, _, attrs in G.edges:
            if (u in nodeset) and (v in nodeset):
                i, j = index[u], index[v]
                e_weight = attrs.get(weight, 1)
                A[i, j] = op([e_weight, A[i, j]])
                if undirected:
                    A[j, i] = A[i, j]
    else:
        # Graph or DiGraph, this is much faster than above
        A = np.full((nlen, nlen), np.nan, order=order)
        for u, nbrdict in G.adj.items():
            for v, d in nbrdict.items():
                try:
                    A[index[u], index[v]] = d.get(weight, 1)
                except KeyError:
                    # This occurs when there are fewer desired nodes than
                    # there are nodes in the graph: len(nodelist) < len(G)
                    pass

    A[np.isnan(A)] = nonedge
    A = np.asarray(A, dtype=dtype)
    return A
