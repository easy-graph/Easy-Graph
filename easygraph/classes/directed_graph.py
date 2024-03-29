from typing import Dict
from typing import List

import easygraph
import easygraph.convert as convert

from easygraph.classes.graph import Graph
from easygraph.utils.exception import EasyGraphError


class DiGraph(Graph):
    """
    Base class for directed graphs.

        Nodes are allowed for any hashable Python objects, including int, string, dict, etc.
        Edges are stored as Python dict type, with optional key/value attributes.

    Parameters
    ----------
    graph_attr : keywords arguments, optional (default : None)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    Graph

    Examples
    --------
    Create an empty directed graph with no nodes and edges.

    >>> G = eg.DiGraph()

    Create a deep copy graph *G2* from existing Graph *G1*.

    >>> G2 = G1.copy()

    Create an graph with attributes.

    >>> G = eg.DiGraph(name='Karate Club', date='2020.08.21')

    **Attributes:**

    Returns the adjacency matrix of the graph.

    >>> G.adj

    Returns all the nodes with their attributes.

    >>> G.nodes

    Returns all the edges with their attributes.

    >>> G.edges

    """

    gnn_data_dict_factory = dict
    graph_attr_dict_factory = dict
    node_dict_factory = dict
    node_attr_dict_factory = dict
    adjlist_outer_dict_factory = dict
    adjlist_inner_dict_factory = dict
    edge_attr_dict_factory = dict
    node_index_dict = dict

    def __init__(self, incoming_graph_data=None, **graph_attr):
        self.graph = self.graph_attr_dict_factory()
        self._ndata = self.gnn_data_dict_factory()
        self._node = self.node_dict_factory()
        self._adj = self.adjlist_outer_dict_factory()
        self._pred = self.adjlist_outer_dict_factory()
        self._node_index = self.node_index_dict()
        self._id = 0
        self.cflag = 0
        self.cache = {}
        self._node_index = self.node_index_dict()
        if incoming_graph_data is not None:
            convert.to_easygraph_graph(incoming_graph_data, create_using=self)
        self.graph.update(graph_attr)

    def __iter__(self):
        return iter(self._node)

    def __len__(self):
        return len(self._node)

    def __contains__(self, node):
        try:
            return node in self._node
        except TypeError:
            return False

    def __getitem__(self, node):
        # return list(self._adj[node].keys())
        return self._adj[node]

    @property
    def node_index(self):
        return self._node_index

    @property
    def ndata(self):
        return self._ndata

    @property
    def pred(self):
        """
        Return the pred of each node
        """
        return self._pred

    @property
    def adj(self):
        """
        Return the adjacency matrix
        """
        return self._adj

    @property
    def nodes(self):
        """
        return [node for node in self._node]
        """
        return self._node

    @property
    def edges(self):
        """
        Return an edge list
        """
        edges = list()
        for u in self._adj:
            for v in self._adj[u]:
                edges.append((u, v, self._adj[u][v]))
        return edges

    @property
    def name(self):
        """String identifier of the graph.

        This graph attribute appears in the attribute dict G.graph
        keyed by the string `"name"`. as well as an attribute (technically
        a property) `G.name`. This is entirely user controlled.
        """
        return self.graph.get("name", "")

    @name.setter
    def name(self, s):
        """
        Set graph name

        Parameters
        ----------
        s : name
        """
        self.graph["name"] = s

    @property
    def node_index(self):
        """
        Assign an integer index for each node (start from 0)
        """
        if self.cache.get("node_index", None) is None:
            node2index_dict = {}
            index = 0
            for n in self.nodes:
                node2index_dict[n] = index
                index += 1
            self.cache["node_index"] = node2index_dict
        return self.cache["node_index"]

    @property
    def index2node(self):
        """
        Assign an integer index for each node (start from 0)
        """
        if self.cache.get("index2node", None) is None:
            index2node_dict = {}
            for index, n in enumerate(self.nodes):
                index2node_dict[index] = n
            self.cache["index2node"] = index2node_dict
        return self.cache["index2node"]

    def out_degree(self, weight="weight"):
        """Returns the weighted out degree of each node.

        Parameters
        ----------
        weight : string, optional (default : 'weight')
            Weight key of the original weighted graph.

        Returns
        -------
        out_degree : dict
            Each node's (key) weighted out degree (value).

        Notes
        -----
        If the graph is not weighted, all the weights will be regarded as 1.

        See Also
        --------
        in_degree
        degree

        Examples
        --------

        >>> G.out_degree(weight='weight')

        """
        degree = dict()
        for u, v, d in self.edges:
            if u in degree:
                degree[u] += d.get(weight, 1)
            else:
                degree[u] = d.get(weight, 1)

        # For isolated nodes
        for node in self.nodes:
            if node not in degree:
                degree[node] = 0

        return degree

    def in_degree(self, weight="weight"):
        """Returns the weighted in degree of each node.

        Parameters
        ----------
        weight : string, optional (default : 'weight')
            Weight key of the original weighted graph.

        Returns
        -------
        in_degree : dict
            Each node's (key) weighted in degree (value).

        Notes
        -----
        If the graph is not weighted, all the weights will be regarded as 1.

        See Also
        --------
        out_degree
        degree

        Examples
        --------

        >>> G.in_degree(weight='weight')

        """
        degree = dict()
        for u, v, d in self.edges:
            if v in degree:
                degree[v] += d.get(weight, 1)
            else:
                degree[v] = d.get(weight, 1)

        # For isolated nodes
        for node in self.nodes:
            if node not in degree:
                degree[node] = 0

        return degree

    def degree(self, weight="weight"):
        """Returns the weighted degree of each node, i.e. sum of out/in degree.

        Parameters
        ----------
        weight : string, optional (default : 'weight')
            Weight key of the original weighted graph.

        Returns
        -------
        degree : dict
            Each node's (key) weighted in degree (value).
            For directed graph, it returns the sum of out degree and in degree.

        Notes
        -----
        If the graph is not weighted, all the weights will be regarded as 1.

        See also
        --------
        out_degree
        in_degree

        Examples
        --------

        >>> G.degree()
        >>> G.degree(weight='weight')

        or you can customize the weight key

        >>> G.degree(weight='weight_1')

        """
        degree = dict()
        outdegree = self.out_degree(weight=weight)
        indegree = self.in_degree(weight=weight)
        all_nodes = set(outdegree.keys()) | set(indegree.keys())
        for u in all_nodes:
            degree[u] = outdegree[u] + indegree[u]
        return degree

    def size(self, weight=None):
        """Returns the number of edges or total of all edge weights.

        Parameters
        -----------
        weight : String or None, optional
            The weight key. If None, it will calculate the number of
            edges, instead of total of all edge weights.

        Returns
        -------
        size : int or float, optional (default: None)
            The number of edges or total of all edge weights.

        Examples
        --------

        Returns the number of edges in G:

        >>> G.size()

        Returns the total of all edge weights in G:

        >>> G.size(weight='weight')

        """
        s = sum(d for v, d in self.out_degree(weight=weight).items())
        return int(s) if weight is None else s

    def number_of_edges(self, u=None, v=None):
        """Returns the number of edges between two nodes.

        Parameters
        ----------
        u, v : nodes, optional (default=all edges)
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
        For undirected graphs, this method counts the total number of
        edges in the graph:

        >>> G = eg.path_graph(4)
        >>> G.number_of_edges()
        3

        If you specify two nodes, this counts the total number of edges
        joining the two nodes:

        >>> G.number_of_edges(0, 1)
        1

        For directed graphs, this method can count the total number of
        directed edges from `u` to `v`:

        >>> G = eg.DiGraph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(1, 0)
        >>> G.number_of_edges(0, 1)
        1

        """
        if u is None:
            return int(self.size())
        if v in self._adj[u]:
            return 1
        return 0

    def nbunch_iter(self, nbunch=None):
        """Returns an iterator over nodes contained in nbunch that are
        also in the graph.

        The nodes in nbunch are checked for membership in the graph
        and if not are silently ignored.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        Returns
        -------
        niter : iterator
            An iterator over nodes in nbunch that are also in the graph.
            If nbunch is None, iterate over all nodes in the graph.

        Raises
        ------
        EasyGraphError
            If nbunch is not a node or sequence of nodes.
            If a node in nbunch is not hashable.

        See Also
        --------
        Graph.__iter__

        Notes
        -----
        When nbunch is an iterator, the returned iterator yields values
        directly from nbunch, becoming exhausted when nbunch is exhausted.

        To test whether nbunch is a single node, one can use
        "if nbunch in self:", even after processing with this routine.

        If nbunch is not a node or a (possibly empty) sequence/iterator
        or None, a :exc:`EasyGraphError` is raised.  Also, if any object in
        nbunch is not hashable, a :exc:`EasyGraphError` is raised.
        """
        if nbunch is None:  # include all nodes via iterator
            bunch = iter(self._adj)
        elif nbunch in self:  # if nbunch is a single node
            bunch = iter([nbunch])
        else:  # if nbunch is a sequence of nodes

            def bunch_iter(nlist, adj):
                try:
                    for n in nlist:
                        if n in adj:
                            yield n
                except TypeError as err:
                    exc, message = err, err.args[0]
                    # capture error for non-sequence/iterator nbunch.
                    if "iter" in message:
                        exc = EasyGraphError(
                            "nbunch is not a node or a sequence of nodes."
                        )
                    # capture error for unhashable node.
                    if "hashable" in message:
                        exc = EasyGraphError(
                            f"Node {n} in sequence nbunch is not a valid node."
                        )
                    raise exc

            bunch = bunch_iter(nbunch, self._adj)
        return bunch

    def neighbors(self, node):
        """Returns an iterator of a node's neighbors (successors).

        Parameters
        ----------
        node : Hashable
            The target node.

        Returns
        -------
        neighbors : iterator
            An iterator of a node's neighbors (successors).

        Examples
        --------
        >>> G = eg.Graph()
        >>> G.add_edges([(1,2), (2,3), (2,4)])
        >>> for neighbor in G.neighbors(node=2):
        ...     print(neighbor)

        """
        # successors
        try:
            return iter(self._adj[node])
        except KeyError:
            print("No node {}".format(node))

    successors = neighbors

    def predecessors(self, node):
        """Returns an iterator of a node's neighbors (predecessors).

        Parameters
        ----------
        node : Hashable
            The target node.

        Returns
        -------
        neighbors : iterator
            An iterator of a node's neighbors (predecessors).

        Examples
        --------
        >>> G = eg.Graph()
        >>> G.add_edges([(1,2), (2,3), (2,4)])
        >>> for predecessor in G.predecessors(node=2):
        ...     print(predecessor)

        """
        # predecessors
        try:
            return iter(self._pred[node])
        except KeyError:
            print("No node {}".format(node))

    def all_neighbors(self, node):
        """Returns an iterator of a node's neighbors, including both successors and predecessors.

        Parameters
        ----------
        node : Hashable
            The target node.

        Returns
        -------
        neighbors : iterator
            An iterator of a node's neighbors, including both successors and predecessors.

        Examples
        --------
        >>> G = eg.Graph()
        >>> G.add_edges([(1,2), (2,3), (2,4)])
        >>> for neighbor in G.all_neighbors(node=2):
        ...     print(neighbor)

        """
        # union of successors and predecessors
        try:
            neighbors = list(self._adj[node])
            neighbors.extend(self._pred[node])
            return iter(neighbors)
        except KeyError:
            print("No node {}".format(node))

    def add_node(self, node_for_adding, **node_attr):
        """Add one node

        Add one node, type of which is any hashable Python object, such as int, string, dict, or even Graph itself.
        You can add with node attributes using Python dict type.

        Parameters
        ----------
        node_for_adding : any hashable Python object
            Nodes for adding.

        node_attr : keywords arguments, optional
            The node attributes.
            You can customize them with different key-value pairs.

        See Also
        --------
        add_nodes

        Examples
        --------
        >>> G.add_node('a')
        >>> G.add_node('hello world')
        >>> G.add_node('Jack', age=10)

        >>> G.add_node('Jack', **{
        ...     'age': 10,
        ...     'gender': 'M'
        ... })

        """
        self._add_one_node(node_for_adding, node_attr)

    def add_nodes(self, nodes_for_adding: list, nodes_attr: List[Dict] = []):
        """Add nodes with a list of nodes.

        Parameters
        ----------
        nodes_for_adding : list

        nodes_attr : list of dict
            The corresponding attribute for each of *nodes_for_adding*.

        See Also
        --------
        add_node

        Examples
        --------
        Add nodes with a list of nodes.
        You can add with node attributes using a list of Python dict type,
        each of which is the attribute of each node, respectively.

        >>> G.add_nodes([1, 2, 'a', 'b'])
        >>> G.add_nodes(range(1, 200))

        >>> G.add_nodes(['Jack', 'Tom', 'Lily'], nodes_attr=[
        ...     {
        ...         'age': 10,
        ...         'gender': 'M'
        ...     },
        ...     {
        ...         'age': 11,
        ...         'gender': 'M'
        ...     },
        ...     {
        ...         'age': 10,
        ...         'gender': 'F'
        ...     }
        ... ])

        """
        if nodes_attr is None:
            nodes_attr = []
        if not len(nodes_attr) == 0:  # Nodes attributes included in input
            assert len(nodes_for_adding) == len(
                nodes_attr
            ), "Nodes and Attributes lists must have same length."
        else:  # Set empty attribute for each node
            nodes_attr = [dict() for i in range(len(nodes_for_adding))]

        for i in range(len(nodes_for_adding)):
            try:
                self._add_one_node(nodes_for_adding[i], nodes_attr[i])
            except Exception as err:
                print(err)
                pass

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add multiple nodes.

        Parameters
        ----------
        nodes_for_adding : iterable container
            A container of nodes (list, dict, set, etc.).
            OR
            A container of (node, attribute dict) tuples.
            Node attributes are updated using the attribute dict.
        attr : keyword arguments, optional (default= no attributes)
            Update attributes for all nodes in nodes.
            Node attributes specified in nodes as a tuple take
            precedence over attributes specified via keyword arguments.

        See Also
        --------
        add_node

        Examples
        --------
        >>> G = eg.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_nodes_from("Hello")
        >>> K3 = eg.Graph([(0, 1), (1, 2), (2, 0)])
        >>> G.add_nodes_from(K3)
        >>> sorted(G.nodes(), key=str)
        [0, 1, 2, 'H', 'e', 'l', 'o']

        Use keywords to update specific node attributes for every node.

        >>> G.add_nodes_from([1, 2], size=10)
        >>> G.add_nodes_from([3, 4], weight=0.4)

        Use (node, attrdict) tuples to update attributes for specific nodes.

        >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
        >>> G.nodes[1]["size"]
        11
        >>> H = eg.Graph()
        >>> H.add_nodes_from(G.nodes(data=True))
        >>> H.nodes[1]["size"]
        11

        """
        for n in nodes_for_adding:
            try:
                newnode = n not in self._node
                newdict = attr
            except TypeError:
                n, ndict = n
                newnode = n not in self._node
                newdict = attr.copy()
                newdict.update(ndict)
            if newnode:
                if n is None:
                    raise ValueError("None cannot be a node")
                self._adj[n] = self.adjlist_inner_dict_factory()
                self._pred[n] = self.adjlist_inner_dict_factory()
                self._node[n] = self.node_attr_dict_factory()
            self._node[n].update(newdict)

    def _add_one_node(self, one_node_for_adding, node_attr: dict = {}):
        node = one_node_for_adding
        if node not in self._node:
            self._node_index[node] = self._id
            self._id += 1
            self._adj[node] = self.adjlist_inner_dict_factory()
            self._pred[node] = self.adjlist_inner_dict_factory()
            attr_dict = self._node[node] = self.node_attr_dict_factory()
            attr_dict.update(node_attr)
        else:  # If already exists, there is no complain and still updating the node attribute
            self._node[node].update(node_attr)

    def add_edge(self, u_of_edge, v_of_edge, **edge_attr):
        """Add a directed edge.

        Parameters
        ----------
        u_of_edge : object
            The start end of this edge

        v_of_edge : object
            The destination end of this edge

        edge_attr : keywords arguments, optional
            The attribute of the edge.

        Notes
        -----
        Nodes of this edge will be automatically added to the graph, if they do not exist.

        See Also
        --------
        add_edges

        Examples
        --------

        >>> G.add_edge(1,2)
        >>> G.add_edge('Jack', 'Tom', weight=10)

        Add edge with attributes, edge weight, for example,

        >>> G.add_edge(1, 2, **{
        ...     'weight': 20
        ... })

        """
        self._add_one_edge(u_of_edge, v_of_edge, edge_attr)

    def add_weighted_edge(self, u_of_edge, v_of_edge, weight):
        """Add a weighted edge

        Parameters
        ----------
        u_of_edge : start node

        v_of_edge : end node

        weight : weight value

        Examples
        --------
        Add a weighted edge

        >>> G.add_weighted_edge( 1 , 3 , 1.0)

        """
        self._add_one_edge(u_of_edge, v_of_edge, edge_attr={"weight": weight})

    def add_edges(self, edges_for_adding, edges_attr: List[Dict] = []):
        """Add a list of edges.

        Parameters
        ----------
        edges_for_adding : list of 2-element tuple
            The edges for adding. Each element is a (u, v) tuple, and u, v are
            start end and destination end, respectively.

        edges_attr : list of dict, optional
            The corresponding attributes for each edge in *edges_for_adding*.

        Examples
        --------
        Add a list of edges into *G*

        >>> G.add_edges([
        ...     (1, 2),
        ...     (3, 4),
        ...     ('Jack', 'Tom')
        ... ])

        Add edge with attributes, for example, edge weight,

        >>> G.add_edges([(1,2), (2, 3)], edges_attr=[
        ...     {
        ...         'weight': 20
        ...     },
        ...     {
        ...         'weight': 15
        ...     }
        ... ])

        """
        if edges_attr is None:
            edges_attr = []
        if not len(edges_attr) == 0:  # Edges attributes included in input
            assert len(edges_for_adding) == len(
                edges_attr
            ), "Edges and Attributes lists must have same length."
        else:  # Set empty attribute for each edge
            edges_attr = [dict() for i in range(len(edges_for_adding))]

        for i in range(len(edges_for_adding)):
            try:
                edge = edges_for_adding[i]
                attr = edges_attr[i]
                assert len(edge) == 2, "Edge tuple {} must be 2-tuple.".format(edge)
                self._add_one_edge(edge[0], edge[1], attr)
            except Exception as err:
                print(err)

    def add_edges_from(self, ebunch_to_add, **attr):
        """Add all the edges in ebunch_to_add.

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the container will be added to the
            graph. The edges must be given as 2-tuples (u, v) or
            3-tuples (u, v, d) where d is a dictionary containing edge data.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

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
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}
            else:
                raise EasyGraphError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
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
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            self._adj[u][v] = datadict
            self._pred[v][u] = datadict

    def add_edges_from_file(self, file, weighted=False):
        """Added edges from file
        For example, txt files,

        Each line is in form like:
        a b 23.0
        which denotes an edge `a → b` with weight 23.0.

        Parameters
        ----------
        file : string
            The file path.

        weighted : boolean, optional (default : False)
            If the file consists of weight information, set `True`.
            The weight key will be set as 'weight'.

        Examples
        --------

        If `./club_network.txt` is:

        Jack Mary 23.0

        Mary Tom 15.0

        Tom Ben 20.0

        Then add them to *G*

        >>> G.add_edges_from_file(file='./club_network.txt', weighted=True)


        """
        import re

        with open(file, "r") as fp:
            edges = fp.readlines()
        if weighted:
            for edge in edges:
                edge = re.sub(",", " ", edge)
                edge = edge.split()
                try:
                    self.add_edge(edge[0], edge[1], weight=float(edge[2]))
                except:
                    pass
        else:
            for edge in edges:
                edge = re.sub(",", " ", edge)
                edge = edge.split()
                try:
                    self.add_edge(edge[0], edge[1])
                except:
                    pass

    def _add_one_edge(self, u_of_edge, v_of_edge, edge_attr: dict = {}):
        u, v = u_of_edge, v_of_edge
        # add nodes
        if u not in self._node:
            self._add_one_node(u)
        if v not in self._node:
            self._add_one_node(v)
        # add the edge
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(edge_attr)
        self._adj[u][v] = datadict
        self._pred[v][u] = datadict

    def remove_node(self, node_to_remove):
        """Remove one node from your graph.

        Parameters
        ----------
        node_to_remove : object
            The node you want to remove.

        See Also
        --------
        remove_nodes

        Examples
        --------
        Remove node *Jack* from *G*

        >>> G.remove_node('Jack')

        """
        try:
            succs = list(self._adj[node_to_remove])
            preds = list(self._pred[node_to_remove])
            del self._node[node_to_remove]
        except KeyError:  # Node not exists in self
            raise KeyError("No node {} in graph.".format(node_to_remove))
        for succ in succs:  # Remove edges start with node_to_remove
            del self._pred[succ][node_to_remove]
        for pred in preds:  # Remove edges end with node_to_remove
            del self._adj[pred][node_to_remove]

        # Remove this node
        del self._adj[node_to_remove]
        del self._pred[node_to_remove]

    def remove_nodes(self, nodes_to_remove: list):
        """Remove nodes from your graph.

        Parameters
        ----------
        nodes_to_remove : list of object
            The list of nodes you want to remove.

        See Also
        --------
        remove_node

        Examples
        --------
        Remove node *[1, 2, 'a', 'b']* from *G*

        >>> G.remove_nodes([1, 2, 'a', 'b'])

        """
        for (
            node
        ) in (
            nodes_to_remove
        ):  # If not all nodes included in graph, give up removing other nodes
            assert node in self._node, "Remove Error: No node {} in graph".format(node)
        for node in nodes_to_remove:
            self.remove_node(node)

    def remove_edge(self, u, v):
        """Remove one edge from your graph.

        Parameters
        ----------
        u : object
            The start end of the edge.

        v : object
            The destination end of the edge.

        See Also
        --------
        remove_edges

        Examples
        --------
        Remove edge (1,2) from *G*

        >>> G.remove_edge(1,2)

        """
        try:
            del self._adj[u][v]
            del self._pred[v][u]
        except KeyError:
            raise KeyError("No edge {}-{} in graph.".format(u, v))

    def remove_edges(self, edges_to_remove: [tuple]):
        """Remove a list of edges from your graph.

        Parameters
        ----------
        edges_to_remove : list of tuple
            The list of edges you want to remove,
            Each element is (u, v) tuple, which denote the start and destination
            end of the edge, respectively.

        See Also
        --------
        remove_edge

        Examples
        --------
        Remove the edges *('Jack', 'Mary')* amd *('Mary', 'Tom')* from *G*

        >>> G.remove_edge([
        ...     ('Jack', 'Mary'),
        ...     ('Mary', 'Tom')
        ... ])

        """
        for edge in edges_to_remove:
            u, v = edge[:2]
            self.remove_edge(u, v)

    def remove_edges_from(self, ebunch):
        """Remove all edges specified in ebunch.

        Parameters
        ----------
        ebunch: list or container of edge tuples
            Each edge given in the list or container will be removed
            from the graph. The edges can be:

                - 2-tuples (u, v) edge between u and v.
                - 3-tuples (u, v, k) where k is ignored.

        See Also
        --------
        remove_edge : remove a single edge

        Notes
        -----
        Will fail silently if an edge in ebunch is not in the graph.

        Examples
        --------
        >>> G = eg.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> ebunch = [(1, 2), (2, 3)]
        >>> G.remove_edges_from(ebunch)
        """
        for e in ebunch:
            u, v = e[:2]  # ignore edge data
            if u in self._adj and v in self._adj[u]:
                del self._adj[u][v]
                del self._pred[v][u]

    def has_node(self, node):
        """Returns whether a node exists

        Parameters
        ----------
        node

        Returns
        -------
        Bool : True (exist) or False (not exists)

        """
        return node in self._node

    def has_edge(self, u, v):
        """Returns whether an edge exists

        Parameters
        ----------
        u : start node

        v: end node

        Returns
        -------
        Bool : True (exist) or False (not exists)

        """
        try:
            return v in self._adj[u]
        except KeyError:
            return False

    def number_of_nodes(self):
        """Returns the number of nodes.

        Returns
        -------
        number_of_nodes : int
            The number of nodes.
        """
        return len(self._node)

    def is_directed(self):
        """Returns True if graph is a directed_graph, False otherwise."""
        return True

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return False

    def copy(self):
        """Return a deep copy of the graph.

        Returns
        -------
        copy : easygraph.DiGraph
            A deep copy of the original graph.

        Examples
        --------
        *G2* is a deep copy of *G1*

        >>> G2 = G1.copy()

        """
        G = self.__class__()
        G.graph.update(self.graph)
        for node, node_attr in self._node.items():
            G.add_node(node, **node_attr)
        for u, nbrs in self._adj.items():
            for v, edge_data in nbrs.items():
                G.add_edge(u, v, **edge_data)

        return G

    def nodes_subgraph(self, from_nodes: list):
        """Returns a subgraph of some nodes

        Parameters
        ----------
        from_nodes : list of object
            The nodes in subgraph.

        Returns
        -------
        nodes_subgraph : easygraph.Graph
            The subgraph consisting of *from_nodes*.

        Examples
        --------

        >>> G = eg.Graph()
        >>> G.add_edges([(1,2), (2,3), (2,4), (4,5)])
        >>> G_sub = G.nodes_subgraph(from_nodes= [1,2,3])

        """
        # Edge
        from_nodes = set(from_nodes)
        G = self.__class__()
        G.graph.update(self.graph)
        from_nodes = set(from_nodes)
        for node in from_nodes:
            try:
                G.add_node(node, **self._node[node])
            except KeyError:
                pass

            for v, edge_data in self._adj[node].items():
                if v in from_nodes:
                    G.add_edge(node, v, **edge_data)
        return G

    def ego_subgraph(self, center):
        """Returns an ego network graph of a node.

        Parameters
        ----------
        center : object
            The center node of the ego network graph

        Returns
        -------
        ego_subgraph : easygraph.Graph
            The ego network graph of *center*.


        Examples
        --------
        >>> G = eg.Graph()
        >>> G.add_edges([
        ...     ('Jack', 'Maria'),
        ...     ('Maria', 'Andy'),
        ...     ('Jack', 'Tom')
        ... ])
        >>> G.ego_subgraph(center='Jack')
        """
        neighbors_of_center = list(self.all_neighbors(center))
        neighbors_of_center.append(center)
        return self.nodes_subgraph(from_nodes=neighbors_of_center)

    def to_index_node_graph(self, begin_index=0):
        """Returns a deep copy of graph, with each node switched to its index.

        Considering that the nodes of your graph may be any possible hashable Python object,
        you can get an isomorphic graph of the original one, with each node switched to its index.

        Parameters
        ----------
        begin_index : int
            The begin index of the index graph.

        Returns
        -------
        G : easygraph.Graph
            Deep copy of graph, with each node switched to its index.

        index_of_node : dict
            Index of node

        node_of_index : dict
            Node of index

        Examples
        --------
        The following method returns this isomorphic graph and index-to-node dictionary
        as well as node-to-index dictionary.

        >>> G = eg.Graph()
        >>> G.add_edges([
        ...     ('Jack', 'Maria'),
        ...     ('Maria', 'Andy'),
        ...     ('Jack', 'Tom')
        ... ])
        >>> G_index_graph, index_of_node, node_of_index = G.to_index_node_graph()

        """
        G = self.__class__()
        G.graph.update(self.graph)
        index_of_node = dict()
        node_of_index = dict()
        for index, (node, node_attr) in enumerate(self._node.items()):
            G.add_node(index + begin_index, **node_attr)
            index_of_node[node] = index + begin_index
            node_of_index[index + begin_index] = node
        for u, nbrs in self._adj.items():
            for v, edge_data in nbrs.items():
                G.add_edge(index_of_node[u], index_of_node[v], **edge_data)

        return G, index_of_node, node_of_index

    def cpp(self):
        G = DiGraphC()
        G.graph.update(self.graph)
        for u, attr in self.nodes.items():
            G.add_node(u, **attr)
        for u, v, attr in self.edges:
            G.add_edge(u, v, **attr)
        G.generate_linkgraph()
        return G


try:
    import cpp_easygraph

    class DiGraphC(cpp_easygraph.DiGraph):
        cflag = 1

except ImportError:

    class DiGraphC:
        def __init__(self, **graph_attr):
            print(
                "Object cannot be instantiated because C extension has not been"
                " successfully compiled and installed. Please refer to"
                " https://github.com/easy-graph/Easy-Graph/blob/master/README.rst and"
                " reinstall easygraph."
            )
            raise RuntimeError
