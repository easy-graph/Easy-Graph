from copy import deepcopy
from typing import Dict
from typing import List

import easygraph as eg
import easygraph.convert as convert

from easygraph.classes.directed_graph import DiGraph
from easygraph.classes.multigraph import MultiGraph
from easygraph.utils.exception import EasyGraphError


__all__ = ["MultiDiGraph"]


class MultiDiGraph(MultiGraph, DiGraph):
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
        # multigraph_input can be None/True/False. So check "is not False"
        if isinstance(incoming_graph_data, dict) and multigraph_input is not False:
            DiGraph.__init__(self)
            try:
                convert.from_dict_of_dicts(
                    incoming_graph_data, create_using=self, multigraph_input=True
                )
                self.graph.update(attr)
            except Exception as err:
                if multigraph_input is True:
                    raise EasyGraphError(
                        f"converting multigraph_input raised:\n{type(err)}: {err}"
                    )
                DiGraph.__init__(self, incoming_graph_data, **attr)
        else:
            DiGraph.__init__(self, incoming_graph_data, **attr)

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

        >>> G = eg.MultiDiGraph()
        >>> e = (1, 2)
        >>> key = G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        1
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
        [2]

        Associate data to edges using keywords:

        >>> key = G.add_edge(1, 2, weight=3)
        >>> key = G.add_edge(1, 2, key=0, weight=4)  # update data for key=0
        >>> key = G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        For non-string attribute keys, use subscript notation.

        >>> ekey = G.add_edge(1, 2)
        >>> G[1][2][0].update({0: 5})
        >>> G.edges[1, 2, 0].update({0: 5})

        >>>
        >>>
        """
        u, v = u_for_edge, v_for_edge
        if "attr" in attr:
            temp = attr.get("attr")
            attr = temp if temp != None else {}
        # add nodes
        if u not in self._adj:
            if u is None:
                raise ValueError("None cannot be a node")
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._pred[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._adj:
            if v is None:
                raise ValueError("None cannot be a node")
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._pred[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        if key is None:
            key = self.new_edge_key(u, v)
        if v in self._adj[u]:
            keydict = self._adj[u][v]
            datadict = keydict.get(key, self.edge_key_dict_factory())
            datadict.update(attr)
            keydict[key] = datadict
        else:
            # selfloops work this way without special treatment
            datadict = self.edge_attr_dict_factory()
            datadict.update(attr)
            keydict = self.edge_key_dict_factory()
            keydict[key] = datadict
            self._adj[u][v] = keydict
            self._pred[v][u] = keydict
        return key

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
        >>> G = eg.MultiDiGraph()
        >>> G.add_edges_from([(1, 2), (1, 2), (1, 2)])  # key_list returned
        [0, 1, 2]
        >>> G.remove_edge(1, 2)  # remove a single (arbitrary) edge

        For edges with keys

        >>> G = eg.MultiDiGraph()
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
            del self._pred[v][u]

    @property
    def edges(self):
        edges = list()
        for n, nbrs in self._adj.items():
            for nbr, kd in nbrs.items():
                for k, dd in kd.items():
                    edges.append((n, nbr, k, dd))
        return edges

    out_edges = edges

    @property
    def in_edges(self):
        edges = list()
        for n, nbrs in self._adj.items():
            for nbr, kd in nbrs.items():
                for k, dd in kd.items():
                    edges.append((nbr, n, k))
        return edges

    @property
    def degree(self, weight="weight"):
        degree = dict()
        if weight is None:
            for n in self._node:
                succs = self._adj[n]
                preds = self._pred[n]
                deg = sum(len(keys) for keys in succs.values()) + sum(
                    len(keys) for keys in preds.values()
                )
                degree[n] = deg
        else:
            for n in self._node:
                succs = self._adj[n]
                preds = self._pred[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in succs.values()
                    for d in key_dict.values()
                ) + sum(
                    d.get(weight, 1)
                    for key_dict in preds.values()
                    for d in key_dict.values()
                )
                degree[n] = deg

    @property
    def in_degree(self, weight="weight"):
        degree = dict()
        if weight is None:
            for n in self._node:
                preds = self._pred[n]
                deg = sum(len(keys) for keys in preds.values())
                degree[n] = deg
        else:
            for n in self._node:
                preds = self._pred[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in preds.values()
                    for d in key_dict.values()
                )
                degree[n] = deg

    @property
    def out_degree(self, weight="weight"):
        degree = dict()
        if weight is None:
            for n in self._node:
                succs = self._adj[n]
                deg = sum(len(keys) for keys in succs.values())
                degree[n] = deg
        else:
            for n in self._node:
                succs = self._adj[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in succs.values()
                    for d in key_dict.values()
                )
                degree[n] = deg

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return True

    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return True

    def to_undirected(self, reciprocal=False):
        """Returns an undirected representation of the multidigraph.

        Parameters
        ----------
        reciprocal : bool (optional)
          If True only keep edges that appear in both directions
          in the original digraph.

        Returns
        -------
        G : MultiGraph
            An undirected graph with the same name and nodes and
            with edge (u, v, data) if either (u, v, data) or (v, u, data)
            is in the digraph.  If both edges exist in digraph and
            their edge data is different, only one edge is created
            with an arbitrary choice of which edge data to use.
            You must check and correct for this manually if desired.

        See Also
        --------
        MultiGraph, add_edge, add_edges_from

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar D=MultiDiGraph(G) which
        returns a shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed MultiDiGraph to use dict-like
        objects in the data structure, those changes do not transfer
        to the MultiGraph created by this method.

        Examples
        --------
        >>> G = eg.path_graph(2)  # or MultiGraph, etc
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]
        >>> G2 = H.to_undirected()
        >>> list(G2.edges)
        [(0, 1)]
        """
        G = eg.MultiGraph()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        if reciprocal is True:
            G.add_edges_from(
                (u, v, key, deepcopy(data))
                for u, nbrs in self._adj.items()
                for v, keydict in nbrs.items()
                for key, data in keydict.items()
                if v in self._pred[u] and key in self._pred[u][v]
            )
        else:
            G.add_edges_from(
                (u, v, key, deepcopy(data))
                for u, nbrs in self._adj.items()
                for v, keydict in nbrs.items()
                for key, data in keydict.items()
            )
        return G

    def reverse(self, copy=True):
        """Returns the reverse of the graph.

        The reverse is a graph with the same nodes and edges
        but with the directions of the edges reversed.

        Parameters
        ----------
        copy : bool optional (default=True)
            If True, return a new DiGraph holding the reversed edges.
            If False, the reverse graph is created using a view of
            the original graph.
        """
        if copy:
            H = self.__class__()
            H.graph.update(deepcopy(self.graph))
            H.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
            H.add_edges_from((v, u, k, deepcopy(d)) for u, v, k, d in self.edges)
            return H
        return eg.graphviews.reverse_view(self)
