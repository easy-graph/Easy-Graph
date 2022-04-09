import easygraph as eg
from collections.abc import Collection, Generator, Iterator
import warnings

__all__ = [
    "from_dict_of_dicts", "to_easygraph_graph", "from_edgelist",
    "from_dict_of_lists"
]


def to_easygraph_graph(data, create_using=None, multigraph_input=False):
    """Make a EasyGraph graph from a known data structure.

    The preferred way to call this is automatically
    from the class constructor

    >>> d = {0: {1: {"weight": 1}}}  # dict-of-dicts single edge (0,1)
    >>> G = eg.Graph(d)

    instead of the equivalent

    >>> G = eg.from_dict_of_dicts(d)

    Parameters
    ----------
    data : object to be converted

        Current known types are:
         any EasyGraph graph
         dict-of-dicts
         dict-of-lists
         container (e.g. set, list, tuple) of edges
         iterator (e.g. itertools.chain) that produces edges
         generator of edges
         Pandas DataFrame (row per edge)
         numpy matrix
         numpy ndarray
         scipy sparse matrix
         pygraphviz agraph

    create_using : EasyGraph graph constructor, optional (default=eg.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    multigraph_input : bool (default False)
        If True and  data is a dict_of_dicts,
        try to create a multigraph assuming dict_of_dict_of_lists.
        If data and create_using are both multigraphs then create
        a multigraph from a multigraph.

    """

    # EasyGraph graph type
    if hasattr(data, "adj"):
        try:
            result = from_dict_of_dicts(
                data.adj,
                create_using=create_using,
                multigraph_input=data.is_multigraph(),
            )
            # data.graph should be dict-like
            result.graph.update(data.graph)
            # data.nodes should be dict-like
            # result.add_node_from(data.nodes.items()) possible but
            # for custom node_attr_dict_factory which may be hashable
            # will be unexpected behavior
            for n, dd in data.nodes.items():
                result._node[n].update(dd)
            return result
        except Exception as err:
            raise eg.EasyGraphError(
                "Input is not a correct EasyGraph graph.") from err

    # pygraphviz  agraph
    if hasattr(data, "is_strict"):
        try:
            return eg.from_pyGraphviz_agraph(data, create_using=create_using)
        except Exception as err:
            raise eg.EasyGraphError(
                "Input is not a correct pygraphviz graph.") from err

    # dict of dicts/lists
    if isinstance(data, dict):
        try:
            return from_dict_of_dicts(data,
                                      create_using=create_using,
                                      multigraph_input=multigraph_input)
        except Exception as err:
            if multigraph_input is True:
                raise eg.EasyGraphError(
                    f"converting multigraph_input raised:\n{type(err)}: {err}")
            try:
                return from_dict_of_lists(data, create_using=create_using)
            except Exception as err:
                raise TypeError("Input is not known type.") from err

    # Pandas DataFrame
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            if data.shape[0] == data.shape[1]:
                try:
                    return eg.from_pandas_adjacency(data, create_using=create_using)
                except Exception as err:
                    msg = "Input is not a correct Pandas DataFrame adjacency matrix."
                    raise eg.EasyGraphError(msg) from err
            else:
                try:
                    return eg.from_pandas_edgelist(data, edge_attr=True, create_using=create_using)
                except Exception as err:
                    msg = "Input is not a correct Pandas DataFrame adjacency edge-list."
                    raise eg.EasyGraphError(msg) from err
    except ImportError:
        warnings.warn("pandas not found, skipping conversion test.", ImportWarning)

    # numpy matrix or ndarray
    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            try:
                return eg.from_numpy_array(data, create_using=create_using)
            except Exception as err:
                raise eg.EasyGraphError(
                    "Input is not a correct numpy matrix or array."
                ) from err
    except ImportError:
        warnings.warn("numpy not found, skipping conversion test.", ImportWarning)


    # scipy sparse matrix - any format
    try:
        import scipy

        if hasattr(data, "format"):
            try:
                return eg.from_scipy_sparse_matrix(data, create_using=create_using)
            except Exception as err:
                raise eg.EasyGraphError(
                    "Input is not a correct scipy sparse matrix type."
                ) from err
    except ImportError:
        warnings.warn("scipy not found, skipping conversion test.", ImportWarning)


    # Note: most general check - should remain last in order of execution
    # Includes containers (e.g. list, set, dict, etc.), generators, and
    # iterators (e.g. itertools.chain) of edges

    if isinstance(data, (Collection, Generator, Iterator)):
        try:
            return from_edgelist(data, create_using=create_using)
        except Exception as err:
            raise eg.EasyGraphError("Input is not a valid edge list") from err
            
    raise eg.EasyGraphError("Input is not a known data type for conversion.")


def from_dict_of_lists(d, create_using=None):
    G = eg.empty_graph(0, create_using)
    G.add_nodes_from(d)
    if G.is_multigraph() and not G.is_directed():
        # a dict_of_lists can't show multiedges.  BUT for undirected graphs,
        # each edge shows up twice in the dict_of_lists.
        # So we need to treat this case separately.
        seen = {}
        for node, nbrlist in d.items():
            for nbr in nbrlist:
                if nbr not in seen:
                    G.add_edge(node, nbr)
            seen[node] = 1  # don't allow reverse edge to show up
    else:
        G.add_edges_from(
            ((node, nbr) for node, nbrlist in d.items() for nbr in nbrlist))
    return G


def from_dict_of_dicts(d, create_using=None, multigraph_input=False):
    G = eg.empty_graph(0, create_using)
    G.add_nodes_from(d)
    # does dict d represent a MultiGraph or MultiDiGraph?
    if multigraph_input:
        if G.is_directed():
            if G.is_multigraph():
                G.add_edges_from((u, v, key, data) for u, nbrs in d.items()
                                 for v, datadict in nbrs.items()
                                 for key, data in datadict.items())
            else:
                G.add_edges_from((u, v, data) for u, nbrs in d.items()
                                 for v, datadict in nbrs.items()
                                 for key, data in datadict.items())
        else:  # Undirected
            if G.is_multigraph():
                seen = set()  # don't add both directions of undirected graph
                for u, nbrs in d.items():
                    for v, datadict in nbrs.items():
                        if (u, v) not in seen:
                            G.add_edges_from((u, v, key, data)
                                             for key, data in datadict.items())
                            seen.add((v, u))
            else:
                seen = set()  # don't add both directions of undirected graph
                for u, nbrs in d.items():
                    for v, datadict in nbrs.items():
                        if (u, v) not in seen:
                            G.add_edges_from(
                                (u, v, data) for key, data in datadict.items())
                            seen.add((v, u))

    else:  # not a multigraph to multigraph transfer
        if G.is_multigraph() and not G.is_directed():
            # d can have both representations u-v, v-u in dict.  Only add one.
            # We don't need this check for digraphs since we add both directions,
            # or for Graph() since it is done implicitly (parallel edges not allowed)
            seen = set()
            for u, nbrs in d.items():
                for v, data in nbrs.items():
                    if (u, v) not in seen:
                        G.add_edge(u, v, key=0)
                        G[u][v][0].update(data)
                    seen.add((v, u))
        else:
            G.add_edges_from(((u, v, data) for u, nbrs in d.items()
                              for v, data in nbrs.items()))
    return G


def from_edgelist(edgelist, create_using=None):
    """Returns a graph from a list of edges.

    Parameters
    ----------
    edgelist : list or iterator
      Edge tuples

    create_using : EasyGraph graph constructor, optional (default=eg.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Examples
    --------
    >>> edgelist = [(0, 1)]  # single edge (0,1)
    >>> G = eg.from_edgelist(edgelist)

    or

    >>> G = eg.Graph(edgelist)  # use Graph constructor

    """
    G = eg.empty_graph(0, create_using)
    G.add_edges_from(edgelist)
    return G
