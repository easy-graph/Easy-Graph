"""Base class for MultiGraph."""
from copy import deepcopy
from typing import Dict
from typing import List

import easygraph as eg
import easygraph.convert as convert

from easygraph.classes.graph import Graph
from easygraph.utils.exception import EasyGraphError


__all__ = ["MultiGraph"]


class MultiGraph(Graph):
    edge_key_dict_factory = dict

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        """Initialize a graph with edges, name, or graph attributes.

        Parameters
        ----------
        incoming_graph_data : input graph
            Data to initialize graph.  If incoming_graph_data=None (default)
            an empty graph is created.  The data can be an edge list, or any
            EasyGraph graph object.  If the corresponding optional Python
            packages are installed the data can also be a NumPy matrix
            or 2d ndarray, a SciPy sparse matrix, or a PyGraphviz graph.

        multigraph_input : bool or None (default None)
            Note: Only used when `incoming_graph_data` is a dict.
            If True, `incoming_graph_data` is assumed to be a
            dict-of-dict-of-dict-of-dict structure keyed by
            node to neighbor to edge keys to edge data for multi-edges.
            A EasyGraphError is raised if this is not the case.
            If False, :func:`to_easygraph_graph` is used to try to determine
            the dict's graph data structure as either a dict-of-dict-of-dict
            keyed by node to neighbor to edge data, or a dict-of-iterable
            keyed by node to neighbors.
            If None, the treatment for True is tried, but if it fails,
            the treatment for False is tried.

        attr : keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.

        See Also
        --------
        convert

        Examples
        --------
        >>> G = eg.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G = eg.Graph(name="my graph")
        >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
        >>> G = eg.Graph(e)

        Arbitrary graph attribute pairs (key=value) may be assigned

        >>> G = eg.Graph(e, day="Friday")
        >>> G.graph
        {'day': 'Friday'}

        """
        self.edge_key_dict_factory = self.edge_key_dict_factory
        if isinstance(incoming_graph_data, dict) and multigraph_input is not False:
            Graph.__init__(self)
            try:
                convert.from_dict_of_dicts(
                    incoming_graph_data, create_using=self, multigraph_input=True
                )
                self.graph.update(attr)
            except Exception as err:
                if multigraph_input is True:
                    raise eg.EasyGraphError(
                        f"converting multigraph_input raised:\n{type(err)}: {err}"
                    )
                Graph.__init__(self, incoming_graph_data, **attr)
        else:
            Graph.__init__(self, incoming_graph_data, **attr)

    def new_edge_key(self, u, v):
        """Returns an unused key for edges between nodes `u` and `v`.

        The nodes `u` and `v` do not need to be already in the graph.

        Notes
        -----
        In the standard MultiGraph class the new key is the number of existing
        edges between `u` and `v` (increased if necessary to ensure unused).
        The first edge will have key 0, then 1, etc. If an edge is removed
        further new_edge_keys may not be in this order.

        Parameters
        ----------
        u, v : nodes

        Returns
        -------
        key : int
        """
        try:
            keydict = self._adj[u][v]
        except KeyError:
            return 0
        key = len(keydict)
        while key in keydict:
            key += 1
        return key

    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        """Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

        Parameters
        ----------
        u_for_edge, v_for_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        key : hashable identifier, optional (default=lowest unused integer)
            Used to distinguish multiedges between a pair of nodes.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        The edge key assigned to the edge.

        See Also
        --------
        add_edges_from : add a collection of edges

        Notes
        -----
        To replace/update edge data, use the optional key argument
        to identify a unique edge.  Otherwise a new edge will be created.

        EasyGraph algorithms designed for weighted graphs cannot use
        multigraphs directly because it is not clear how to handle
        multiedge weights.  Convert to Graph using edge attribute
        'weight' to enable weighted graph algorithms.

        Default keys are generated using the method `new_edge_key()`.
        This method can be overridden by subclassing the base class and
        providing a custom `new_edge_key()` method.

        Examples
        --------
        The following all add the edge e=(1, 2) to graph G:

        >>> G = eg.MultiGraph()
        >>> e = (1, 2)
        >>> ekey = G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        1
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
        [2]

        Associate data to edges using keywords:

        >>> ekey = G.add_edge(1, 2, weight=3)
        >>> ekey = G.add_edge(1, 2, key=0, weight=4)  # update data for key=0
        >>> ekey = G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        For non-string attribute keys, use subscript notation.

        >>> ekey = G.add_edge(1, 2)
        >>> G[1][2][0].update({0: 5})
        >>> G.edges[1, 2, 0].update({0: 5})
        """
        u, v = u_for_edge, v_for_edge
        # add nodes
        if u not in self._adj:
            if u is None:
                raise ValueError("None cannot be a node")
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._adj:
            if v is None:
                raise ValueError("None cannot be a node")
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        if key is None:
            key = self.new_edge_key(u, v)
        if v in self._adj[u]:
            keydict = self._adj[u][v]
            datadict = keydict.get(key, self.edge_attr_dict_factory())
            datadict.update(attr)
            keydict[key] = datadict
        else:
            # selfloops work this way without special treatment
            datadict = self.edge_attr_dict_factory()
            datadict.update(attr)
            keydict = self.edge_key_dict_factory()
            keydict[key] = datadict
            self._adj[u][v] = keydict
            self._adj[v][u] = keydict
        return key

    def add_edges_from(self, ebunch_to_add, **attr):
        """Add all the edges in ebunch_to_add.

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the container will be added to the
            graph. The edges can be:

                - 2-tuples (u, v) or
                - 3-tuples (u, v, d) for an edge data dict d, or
                - 3-tuples (u, v, k) for not iterable key k, or
                - 4-tuples (u, v, k, d) for an edge with data and key k

        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        A list of edge keys assigned to the edges in `ebunch`.

        See Also
        --------
        add_edge : add a single edge
        add_weighted_edges_from : convenient way to add weighted edges

        Notes
        -----
        Adding the same edge twice has no effect but any edge data
        will be updated when each duplicate edge is added.

        Edge attributes specified in an ebunch take precedence over
        attributes specified via keyword arguments.

        Default keys are generated using the method ``new_edge_key()``.
        This method can be overridden by subclassing the base class and
        providing a custom ``new_edge_key()`` method.

        Examples
        --------
        >>> G = eg.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
        >>> e = zip(range(0, 3), range(1, 4))
        >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3

        Associate data to edges

        >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
        >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")
        """
        keylist = []
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 4:
                u, v, key, dd = e
            elif ne == 3:
                u, v, dd = e
                key = None
            elif ne == 2:
                u, v = e
                dd = {}
                key = None
            else:
                msg = f"Edge tuple {e} must be a 2-tuple, 3-tuple or 4-tuple."
                raise EasyGraphError(msg)
            ddd = {}
            ddd.update(attr)
            try:
                ddd.update(dd)
            except (TypeError, ValueError):
                if ne != 3:
                    raise
                key = dd  # ne == 3 with 3rd value not dict, must be a key
            key = self.add_edge(u, v, key)
            self[u][v][key].update(ddd)
            keylist.append(key)
        return keylist

    def remove_edge(self, u, v, key=None):
        """Remove an edge between u and v.

        Parameters
        ----------
        u, v : nodes
            Remove an edge between nodes u and v.
        key : hashable identifier, optional (default=None)
            Used to distinguish multiple edges between a pair of nodes.
            If None remove a single (arbitrary) edge between u and v.

        Raises
        ------
        EasyGraphError
            If there is not an edge between u and v, or
            if there is no edge with the specified key.

        See Also
        --------
        remove_edges_from : remove a collection of edges

        Examples
        --------
        For multiple edges

        >>> G = eg.MultiGraph()  # or MultiDiGraph, etc
        >>> G.add_edges_from([(1, 2), (1, 2), (1, 2)])  # key_list returned
        [0, 1, 2]
        >>> G.remove_edge(1, 2)  # remove a single (arbitrary) edge

        For edges with keys

        >>> G = eg.MultiGraph()  # or MultiDiGraph, etc
        >>> G.add_edge(1, 2, key="first")
        'first'
        >>> G.add_edge(1, 2, key="second")
        'second'
        >>> G.remove_edge(1, 2, key="second")

        """
        try:
            d = self._adj[u][v]
        except KeyError as err:
            raise EasyGraphError(f"The edge {u}-{v} is not in the graph.") from err
        # remove the edge with specified data
        if key is None:
            d.popitem()
        else:
            try:
                del d[key]
            except KeyError as err:
                msg = f"The edge {u}-{v} with key {key} is not in the graph."
                raise EasyGraphError(msg) from err
        if len(d) == 0:
            # remove the key entries if last edge
            del self._adj[u][v]
            if u != v:  # check for selfloop
                del self._adj[v][u]

    def remove_edges_from(self, ebunch):
        """Remove all edges specified in ebunch.

        Parameters
        ----------
        ebunch: list or container of edge tuples
            Each edge given in the list or container will be removed
            from the graph. The edges can be:

                - 2-tuples (u, v) All edges between u and v are removed.
                - 3-tuples (u, v, key) The edge identified by key is removed.
                - 4-tuples (u, v, key, data) where data is ignored.

        See Also
        --------
        remove_edge : remove a single edge

        Notes
        -----
        Will fail silently if an edge in ebunch is not in the graph.

        Examples
        --------
        Removing multiple copies of edges

        >>> G = eg.MultiGraph()
        >>> keys = G.add_edges_from([(1, 2), (1, 2), (1, 2)])
        >>> G.remove_edges_from([(1, 2), (1, 2)])
        >>> list(G.edges())
        [(1, 2)]
        >>> G.remove_edges_from([(1, 2), (1, 2)])  # silently ignore extra copy
        >>> list(G.edges)  # now empty graph
        []
        """
        for e in ebunch:
            try:
                self.remove_edge(*e[:3])
            except EasyGraphError:
                pass

    def has_edge(self, u, v, key=None):
        """Returns True if the graph has an edge between nodes u and v.

        This is the same as `v in G[u] or key in G[u][v]`
        without KeyError exceptions.

        Parameters
        ----------
        u, v : nodes
            Nodes can be, for example, strings or numbers.

        key : hashable identifier, optional (default=None)
            If specified return True only if the edge with
            key is found.

        Returns
        -------
        edge_ind : bool
            True if edge is in the graph, False otherwise.

        Examples
        --------
        Can be called either using two nodes u, v, an edge tuple (u, v),
        or an edge tuple (u, v, key).

        >>> G = eg.MultiGraph()  # or MultiDiGraph
        >>> G = eg.complete_graph(4, create_using=eg.MultiDiGraph)
        >>> G.has_edge(0, 1)  # using two nodes
        True
        >>> e = (0, 1)
        >>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
        True
        >>> G.add_edge(0, 1, key="a")
        'a'
        >>> G.has_edge(0, 1, key="a")  # specify key
        True
        >>> e = (0, 1, "a")
        >>> G.has_edge(*e)  # e is a 3-tuple (u, v, 'a')
        True

        The following syntax are equivalent:

        >>> G.has_edge(0, 1)
        True
        >>> 1 in G[0]  # though this gives :exc:`KeyError` if 0 not in G
        True

        """
        try:
            if key is None:
                return v in self._adj[u]
            else:
                return key in self._adj[u][v]
        except KeyError:
            return False

    @property
    def edges(self):
        edges = list()
        seen = {}
        for n, nbrs in self._adj.items():
            for nbr, kd in nbrs.items():
                if nbr not in seen:
                    for k, dd in kd.items():
                        edges.append((n, nbr, k, dd))
            seen[n] = 1
        del seen
        return edges

    def get_edge_data(self, u, v, key=None, default=None):
        """Returns the attribute dictionary associated with edge (u, v).

        This is identical to `G[u][v][key]` except the default is returned
        instead of an exception is the edge doesn't exist.

        Parameters
        ----------
        u, v : nodes

        default :  any Python object (default=None)
            Value to return if the edge (u, v) is not found.

        key : hashable identifier, optional (default=None)
            Return data only for the edge with specified key.

        Returns
        -------
        edge_dict : dictionary
            The edge attribute dictionary.

        Examples
        --------
        >>> G = eg.MultiGraph()  # or MultiDiGraph
        >>> key = G.add_edge(0, 1, key="a", weight=7)
        >>> G[0][1]["a"]  # key='a'
        {'weight': 7}
        >>> G.edges[0, 1, "a"]  # key='a'
        {'weight': 7}

        Warning: we protect the graph data structure by making
        `G.edges` and `G[1][2]` read-only dict-like structures.
        However, you can assign values to attributes in e.g.
        `G.edges[1, 2, 'a']` or `G[1][2]['a']` using an additional
        bracket as shown next. You need to specify all edge info
        to assign to the edge data associated with an edge.

        >>> G[0][1]["a"]["weight"] = 10
        >>> G.edges[0, 1, "a"]["weight"] = 10
        >>> G[0][1]["a"]["weight"]
        10
        >>> G.edges[1, 0, "a"]["weight"]
        10

        >>> G = eg.MultiGraph()  # or MultiDiGraph
        >>> G = eg.complete_graph(4, create_using=eg.MultiDiGraph)
        >>> G.get_edge_data(0, 1)
        {0: {}}
        >>> e = (0, 1)
        >>> G.get_edge_data(*e)  # tuple form
        {0: {}}
        >>> G.get_edge_data("a", "b", default=0)  # edge not in graph, return 0
        0
        """
        try:
            if key is None:
                return self._adj[u][v]
            else:
                return self._adj[u][v][key]
        except KeyError:
            return default

    @property
    def degree(self, weight="weight"):
        degree = dict()
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(len(keys) for keys in nbrs.values()) + (
                    n in nbrs and len(nbrs[n])
                )
                degree[n] = deg
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in nbrs.values()
                    for d in key_dict.values()
                )
                if n in nbrs:
                    deg += sum(d.get(weight, 1) for d in nbrs[n].values())
                degree[n] = deg

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return True

    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return False

    def copy(self):
        """Returns a copy of the graph.

        The copy method by default returns an independent shallow copy
        of the graph and attributes. That is, if an attribute is a
        container, that container is shared by the original an the copy.
        Use Python's `copy.deepcopy` for new containers.

        Notes
        -----
        All copies reproduce the graph structure, but data attributes
        may be handled in different ways. There are four types of copies
        of a graph that people might want.

        Deepcopy -- A "deepcopy" copies the graph structure as well as
        all data attributes and any objects they might contain.
        The entire graph object is new so that changes in the copy
        do not affect the original object. (see Python's copy.deepcopy)

        Data Reference (Shallow) -- For a shallow copy the graph structure
        is copied but the edge, node and graph attribute dicts are
        references to those in the original graph. This saves
        time and memory but could cause confusion if you change an attribute
        in one graph and it changes the attribute in the other.
        EasyGraph does not provide this level of shallow copy.

        Independent Shallow -- This copy creates new independent attribute
        dicts and then does a shallow copy of the attributes. That is, any
        attributes that are containers are shared between the new graph
        and the original. This is exactly what `dict.copy()` provides.
        You can obtain this style copy using:

            >>> G = eg.path_graph(5)
            >>> H = G.copy()
            >>> H = eg.Graph(G)
            >>> H = G.__class__(G)

        Fresh Data -- For fresh data, the graph structure is copied while
        new empty data attribute dicts are created. The resulting graph
        is independent of the original and it has no edge, node or graph
        attributes. Fresh copies are not enabled. Instead use:

            >>> H = G.__class__()
            >>> H.add_nodes_from(G)
            >>> H.add_edges_from(G.edges)

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Returns
        -------
        G : Graph
            A copy of the graph.

        See Also
        --------
        to_directed: return a directed copy of the graph.

        Examples
        --------
        >>> G = eg.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> H = G.copy()

        """
        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, key, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, keydict in nbrs.items()
            for key, datadict in keydict.items()
        )
        return G

    def to_directed(self):
        """Returns a directed representation of the graph.

        Returns
        -------
        G : MultiDiGraph
            A directed graph with the same name, same nodes, and with
            each edge (u, v, data) replaced by two directed edges
            (u, v, data) and (v, u, data).

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar D=DiGraph(G) which returns a
        shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed MultiGraph to use dict-like objects
        in the data structure, those changes do not transfer to the
        MultiDiGraph created by this method.

        Examples
        --------
        >>> G = eg.Graph()  # or MultiGraph, etc
        >>> G.add_edge(0, 1)
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]

        If already directed, return a (deep) copy

        >>> G = eg.DiGraph()  # or MultiDiGraph, etc
        >>> G.add_edge(0, 1)
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1)]
        """
        G = eg.MultiDiGraph()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, key, deepcopy(datadict))
            for u, nbrs in self.adj.items()
            for v, keydict in nbrs.items()
            for key, datadict in keydict.items()
        )
        return G

    def number_of_edges(self, u=None, v=None):
        """Returns the number of edges between two nodes.

        Parameters
        ----------
        u, v : nodes, optional (Gefault=all edges)
            If u and v are specified, return the number of edges between
            u and v. Otherwise return the total number of all edges.

        Returns
        -------
        nedges : int
            The number of edges in the graph.  If nodes `u` and `v` are
            specified return the number of edges between those nodes. If
            the graph is directed, this only returns the number of edges
            from `u` to `v`.

        See Also
        --------
        size

        Examples
        --------
        For undirected multigraphs, this method counts the total number
        of edges in the graph::

            >>> G = eg.MultiGraph()
            >>> G.add_edges_from([(0, 1), (0, 1), (1, 2)])
            [0, 1, 0]
            >>> G.number_of_edges()
            3

        If you specify two nodes, this counts the total number of edges
        joining the two nodes::

            >>> G.number_of_edges(0, 1)
            2

        For directed multigraphs, this method can count the total number
        of directed edges from `u` to `v`::

            >>> G = eg.MultiDiGraph()
            >>> G.add_edges_from([(0, 1), (0, 1), (1, 0)])
            [0, 1, 0]
            >>> G.number_of_edges(0, 1)
            2
            >>> G.number_of_edges(1, 0)
            1

        """
        if u is None:
            return self.size()
        try:
            edgedata = self._adj[u][v]
        except KeyError:
            return 0  # no such edge
        return len(edgedata)
