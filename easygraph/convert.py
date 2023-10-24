import warnings

from collections.abc import Collection
from collections.abc import Generator
from collections.abc import Iterator
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import easygraph as eg

from easygraph.utils.exception import EasyGraphError


if TYPE_CHECKING:
    import dgl
    import networkx as nx
    import torch_geometric

    from easygraph import DiGraph
    from easygraph import Graph

__all__ = [
    "from_dict_of_dicts",
    "to_easygraph_graph",
    "from_edgelist",
    "from_dict_of_lists",
    "from_networkx",
    "from_dgl",
    "from_pyg",
    "to_networkx",
    "to_dgl",
    "to_pyg",
    "dict_to_hypergraph",
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
            raise eg.EasyGraphError("Input is not a correct EasyGraph graph.") from err

    # pygraphviz  agraph
    if hasattr(data, "is_strict"):
        try:
            return eg.from_pyGraphviz_agraph(data, create_using=create_using)
        except Exception as err:
            raise eg.EasyGraphError("Input is not a correct pygraphviz graph.") from err

    # dict of dicts/lists
    if isinstance(data, dict):
        try:
            return from_dict_of_dicts(
                data, create_using=create_using, multigraph_input=multigraph_input
            )
        except Exception as err:
            if multigraph_input is True:
                raise eg.EasyGraphError(
                    f"converting multigraph_input raised:\n{type(err)}: {err}"
                )
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
                    return eg.from_pandas_edgelist(
                        data, edge_attr=True, create_using=create_using
                    )
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
            ((node, nbr) for node, nbrlist in d.items() for nbr in nbrlist)
        )
    return G


def from_dict_of_dicts(d, create_using=None, multigraph_input=False):
    G = eg.empty_graph(0, create_using)
    G.add_nodes_from(d)
    # does dict d represent a MultiGraph or MultiDiGraph?
    if multigraph_input:
        if G.is_directed():
            if G.is_multigraph():
                G.add_edges_from(
                    (u, v, key, data)
                    for u, nbrs in d.items()
                    for v, datadict in nbrs.items()
                    for key, data in datadict.items()
                )
            else:
                G.add_edges_from(
                    (u, v, data)
                    for u, nbrs in d.items()
                    for v, datadict in nbrs.items()
                    for key, data in datadict.items()
                )
        else:  # Undirected
            if G.is_multigraph():
                seen = set()  # don't add both directions of undirected graph
                for u, nbrs in d.items():
                    for v, datadict in nbrs.items():
                        if (u, v) not in seen:
                            G.add_edges_from(
                                (u, v, key, data) for key, data in datadict.items()
                            )
                            seen.add((v, u))
            else:
                seen = set()  # don't add both directions of undirected graph
                for u, nbrs in d.items():
                    for v, datadict in nbrs.items():
                        if (u, v) not in seen:
                            G.add_edges_from(
                                (u, v, data) for key, data in datadict.items()
                            )
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
            G.add_edges_from(
                ((u, v, data) for u, nbrs in d.items() for v, data in nbrs.items())
            )
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


def to_networkx(g: "Union[Graph, DiGraph]") -> "Union[nx.Graph, nx.DiGraph]":
    """Convert an EasyGraph to a NetworkX graph.

    Args:
        g (Union[Graph, DiGraph]): An EasyGraph graph

    Raises:
        ImportError is raised if NetworkX is not installed.

    Returns:
        Union[nx.Graph, nx.DiGraph]: Converted NetworkX graph
    """
    # if load_func_name in di_load_functions_name:
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX not found. Please install it.")
    if g.is_directed():
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # copy attributes
    G.graph = deepcopy(g.graph)

    nodes_with_edges = set()
    for v1, v2, _ in g.edges:
        G.add_edge(v1, v2)
        nodes_with_edges.add(v1)
        nodes_with_edges.add(v2)
    for node in set(g.nodes) - nodes_with_edges:
        G.add_node(node)
    return G


def from_networkx(g: "Union[nx.Graph, nx.DiGraph]") -> "Union[Graph, DiGraph]":
    """Convert a NetworkX graph to an EasyGraph graph.

    Args:
        g (Union[nx.Graph, nx.DiGraph]): A NetworkX graph

    Returns:
        Union[Graph, DiGraph]: Converted EasyGraph graph
    """
    # try:
    #     import networkx as nx
    # except ImportError:
    #     raise ImportError("NetworkX not found. Please install it.")
    if g.is_directed():
        G = eg.DiGraph()
    else:
        G = eg.Graph()

    # copy attributes
    G.graph = deepcopy(g.graph)

    nodes_with_edges = set()
    for v1, v2 in g.edges:
        G.add_edge(v1, v2)
        nodes_with_edges.add(v1)
        nodes_with_edges.add(v2)
    for node in set(g.nodes) - nodes_with_edges:
        G.add_node(node)
    return G


def to_dgl(g: "Union[Graph, DiGraph]"):
    """Convert an EasyGraph graph to a DGL graph.

    Args:
        g (Union[Graph, DiGraph]): An EasyGraph graph

    Raises:
        ImportError: If DGL is not installed.

    Returns:
        DGLGraph: Converted DGL graph
    """
    try:
        import dgl
    except ImportError:
        raise ImportError("DGL not found. Please install it.")
    g_nx = to_networkx(g)
    g_dgl = dgl.from_networkx(g_nx)
    return g_dgl


def from_dgl(g) -> "Union[Graph, DiGraph]":
    """Convert a DGL graph to an EasyGraph graph.

    Args:
        g (DGLGraph): A DGL graph

    Raises:
        ImportError: If DGL is not installed.

    Returns:
        Union[Graph, DiGraph]: Converted EasyGraph graph
    """
    try:
        import dgl
    except ImportError:
        raise ImportError("DGL not found. Please install it.")
    g_nx = dgl.to_networkx(g)
    g_eg = from_networkx(g_nx)
    return g_eg


def to_pyg(
    G: Any,
    group_node_attrs: Optional[Union[List[str], all]] = None,  # type: ignore
    group_edge_attrs: Optional[Union[List[str], all]] = None,  # type: ignore
) -> "torch_geometric.data.Data":  # type: ignore
    r"""Converts a :obj:`easygraph.Graph` or :obj:`easygraph.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (easygraph.Graph or easygraph.DiGraph): A easygraph graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.

    Examples:

        >>> import torch_geometric as pyg

        >>> pyg_to_networkx = pyg.utils.convert.to_networkx  # type: ignore
        >>> networkx_to_pyg = pyg.utils.convert.from_networkx  # type: ignore
        >>> Data = pyg.data.Data  # type: ignore
        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> g = pyg_to_networkx(data)
        >>> # A `Data` object is returned
        >>> to_pyg(g)
        Data(edge_index=[2, 6], num_nodes=4)
    """
    try:
        import torch_geometric as pyg

        pyg_to_networkx = pyg.utils.convert.to_networkx  # type: ignore
        networkx_to_pyg = pyg.utils.convert.from_networkx  # type: ignore
    except ImportError:
        raise ImportError("pytorch_geometric not found. Please install it.")

    g_nx = to_networkx(G)
    g_pyg = networkx_to_pyg(g_nx, group_node_attrs, group_edge_attrs)
    return g_pyg


def from_pyg(
    data: "torch_geometric.data.Data",  # type: ignore
    node_attrs: Optional[Iterable[str]] = None,
    edge_attrs: Optional[Iterable[str]] = None,
    graph_attrs: Optional[Iterable[str]] = None,
    to_undirected: Optional[Union[bool, str]] = False,
    remove_self_loops: bool = False,
) -> Any:
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`easygraph.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`easygraph.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        graph_attrs (iterable of str, optional): The graph attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool or str, optional): If set to :obj:`True` or
            "upper", will return a :obj:`easygraph.Graph` instead of a
            :obj:`easygraph.DiGraph`. The undirected graph will correspond to
            the upper triangle of the corresponding adjacency matrix.
            Similarly, if set to "lower", the undirected graph will correspond
            to the lower triangle of the adjacency matrix. (default:
            :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)

    Examples:

        >>> import torch_geometric as pyg

        >>> Data = pyg.data.Data  # type: ignore
        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> from_pyg(data)
        <easygraph.classes.digraph.DiGraph at 0x2713fdb40d0>

    """

    try:
        import torch_geometric as pyg

        pyg_to_networkx = pyg.utils.convert.to_networkx  # type: ignore
        networkx_to_pyg = pyg.utils.convert.from_networkx  # type: ignore
    except ImportError:
        raise ImportError("pytorch_geometric not found. Please install it.")
    g_nx = pyg_to_networkx(
        data, node_attrs, edge_attrs, graph_attrs, to_undirected, remove_self_loops
    )
    g_eg = from_networkx(g_nx)
    return g_eg


def dict_to_hypergraph(data, max_order=None, is_dynamic=False):
    """
    A function to read a file in a standardized JSON format.

    Parameters
    ----------
    data: dict
        A dictionary in the hypergraph JSON format
    max_order: int, optional
        Maximum order of edges to add to the hypergraph

    Returns
    -------
    A Hypergraph object
        The loaded hypergraph

    Raises
    ------
    EasyGraphError
        If the JSON is not in a format that can be loaded.

    See Also
    --------
    read_json

    """

    timestamp_lst = list()
    node_data = data["node-data"]
    node_num = len(node_data)
    G = eg.Hypergraph(num_v=node_num)
    try:
        # print(len(data["node-data"]))
        for index, dd in data["node-data"].items():
            id = int(index) - 1
            G.v_property[id] = dd
    except KeyError:
        raise EasyGraphError("Failed to import node attributes.")

    # try:
    # import time
    rows = []
    cols = []
    edge_flag_dict = {}
    e_property_dict = data["edge-data"]
    edge_id = 0
    for index, edge in data["edge-dict"].items():
        # print("id:",id)
        if max_order and len(edge) > max_order + 1:
            continue

        try:
            id = int(index)
        except ValueError as e:
            raise TypeError(
                f"Failed to convert the edge with ID {id} to type int."
            ) from e

        try:
            edge = [int(n) - 1 for n in edge]
            if tuple(edge) not in edge_flag_dict:
                edge_flag_dict[tuple(edge)] = 1
                rows.extend(edge)
                cols.extend(len(edge) * [edge_id])
                edge_id += 1

        except ValueError as e:
            raise TypeError(f"Failed to convert nodes to type int.") from e

        if is_dynamic:
            G.add_hyperedges(
                e_list=edge,
                e_property=e_property_dict[str(id)],
                group_name=e_property_dict[str(id)]["timestamp"],
            )

            timestamp_lst.append(e_property_dict[str(id)]["timestamp"])
        else:
            G.add_hyperedges(e_list=edge, e_property=e_property_dict[str(id)])
    G._rows = rows
    G._cols = cols
    return G, timestamp_lst
