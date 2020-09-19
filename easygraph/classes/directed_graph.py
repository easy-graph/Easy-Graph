from copy import deepcopy


class DiGraph(object):
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

    >>> G = eg.Graph()

    Create a deep copy graph *G2* from existing Graph *G1*.

    >>> G2 = G1.copy()

    Create an graph with attributes.

    >>> G = eg.Graph(name='Karate Club', date='2020.08.21')

    **Attributes:**

    Returns the adjacency matrix of the graph.

    >>> G.adj

    Returns all the nodes with their attributes.

    >>> G.nodes

    Returns all the edges with their attributes.

    >>> G.edges

    """
    graph_attr_dict_factory = dict
    node_dict_factory = dict
    node_attr_dict_factory = dict
    adjlist_outer_dict_factory = dict
    adjlist_inner_dict_factory = dict
    edge_attr_dict_factory = dict

    def __init__(self, **graph_attr):
        self.graph = self.graph_attr_dict_factory()
        self._node = self.node_dict_factory()
        self._adj = self.adjlist_outer_dict_factory()
        self._pred = self.adjlist_outer_dict_factory()

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
    def adj(self):
        return self._adj

    @property
    def nodes(self):
        return self._node
        # return [node for node in self._node]

    @property
    def edges(self):
        edges = list()
        for u in self._adj:
            for v in self._adj[u]:
                edges.append((u, v, self._adj[u][v]))
        return edges

    def out_degree(self, weight='weight'):
        """Returns the weighted out degree of each node.

        Parameters
        ----------
        weight : string, optinal (default : 'weight')
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

    def in_degree(self, weight='weight'):
        """Returns the weighted in degree of each node.

        Parameters
        ----------
        weight : string, optinal (default : 'weight')
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

    def degree(self, weight='weight'):
        """Returns the weighted degree of each node, i.e. sum of out/in degree.

        Parameters
        ----------
        weight : string, optinal (default : 'weight')
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
        for u in outdegree:
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

    def neighbors(self, node):
        """Returns an iterator of a node's neighbors (sucessors).

        Parameters
        ----------
        node : object
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
        node : object
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
        node : object
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

    def add_nodes(self, nodes_for_adding: list, nodes_attr: [dict] = []):
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
        if not len(nodes_attr) == 0:  # Nodes attributes included in input
            assert len(nodes_for_adding) == len(
                nodes_attr), "Nodes and Attributes lists must have same length."
        else:  # Set empty attribute for each node
            nodes_attr = [dict() for i in range(len(nodes_for_adding))]

        for i in range(len(nodes_for_adding)):
            try:
                self._add_one_node(nodes_for_adding[i], nodes_attr[i])
            except Exception as err:
                print(err)
                pass

    def _add_one_node(self, one_node_for_adding, node_attr: dict = {}):
        node = one_node_for_adding
        if node not in self._node:
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
        self._add_one_edge(u_of_edge, v_of_edge, edge_attr={"weight": weight})

    def add_edges(self, edges_for_adding, edges_attr: [dict] = []):
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
        if not len(edges_attr) == 0:  # Edges attributes included in input
            assert len(edges_for_adding) == len(
                edges_attr), "Edges and Attributes lists must have same length."
        else:  # Set empty attribute for each edge
            edges_attr = [dict() for i in range(len(edges_for_adding))]

        for i in range(len(edges_for_adding)):
            try:
                edge = edges_for_adding[i]
                attr = edges_attr[i]
                assert len(
                    edge) == 2, "Edge tuple {} must be 2-tuple.".format(edge)
                self._add_one_edge(edge[0], edge[1], attr)
            except Exception as err:
                print(err)

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
            If the file consists of weight infomation, set `True`.
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
        with open(file, 'r') as fp:
            edges = fp.readlines()
        if weighted:
            for edge in edges:
                edge = re.sub(',', ' ', edge)
                edge = edge.split()
                try:
                    self.add_edge(edge[0], edge[1], weight=float(edge[2]))
                except:
                    pass
        else:
            for edge in edges:
                edge = re.sub(',', ' ', edge)
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
        for node in nodes_to_remove:  # If not all nodes included in graph, give up removing other nodes
            assert (node in self._node), "Remove Error: No node {} in graph".format(
                node)
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

    def has_node(self, node):
        return node in self._node

    def has_edge(self, u, v):
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

    def number_of_edges(self):
        """Returns the number of edges.

        Returns
        -------
        number_of_edges : int
            The number of edges.
        """
        return int(self.size())

    def is_directed(self):
        return True

    def copy(self):
        """Return a deep copy of the graph.

        Returns
        -------
        copy : easygraph.DiGraph
            A deep copy of the orginal graph.

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
        G = self.__class__()
        G.graph.update(self.graph)
        for node in from_nodes:
            try:
                G.add_node(node, **self._node[node])
            except KeyError:
                pass

            # Edge
            from_nodes = set(from_nodes)
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
