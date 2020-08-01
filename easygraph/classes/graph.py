from copy import deepcopy


class Graph(object):
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
        seen = set()
        for u in self._adj:
            for v in self._adj[u]:
                if (u, v) not in seen:
                    seen.add((u, v))
                    seen.add((v, u))
                    edges.append((u, v, self._adj[u][v]))
        del seen
        return edges

    def degree(self, weight='weight'):
        degree = dict()
        for u, v, d in self.edges:
            if u in degree:
                degree[u] += d.get(weight, 1)
            else:
                degree[u] = d.get(weight, 1)
            if v in degree:
                degree[v] += d.get(weight, 1)
            else:
                degree[v] = d.get(weight, 1)

        # For isolated nodes
        for node in self.nodes:
            if node not in degree:
                degree[node] = 0

        return degree

    def size(self, weight=None):
        """
        Returns the number of edges or total of all edge weights.

        Parameters
        -----------
        weight : String or None
            key for edge weight.
        """
        s = sum(d for v, d in self.degree(weight=weight).items())
        return s // 2 if weight is None else s / 2

    def neighbors(self, node):
        try:
            return iter(self._adj[node])
        except KeyError:
            print("No node {}".format(node))

    all_neighbors = neighbors

    def add_node(self, node_for_adding, **node_attr):
        self._add_one_node(node_for_adding, node_attr)

    def add_nodes(self, nodes_for_adding: list, nodes_attr: [dict] = []):
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
            attr_dict = self._node[node] = self.node_attr_dict_factory()
            attr_dict.update(node_attr)
        else:  # If already exists, there is no complain and still updating the node attribute
            self._node[node].update(node_attr)

    def add_edge(self, u_of_edge, v_of_edge, **edge_attr):
        self._add_one_edge(u_of_edge, v_of_edge, edge_attr)

    def add_weighted_edge(self, u_of_edge, v_of_edge, weight):
        self._add_one_edge(u_of_edge, v_of_edge, edge_attr={"weight": weight})

    def add_edges(self, edges_for_adding, edges_attr: [dict] = []):
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
        """
        Added edges from file, for example, txt files.
        Each line is in form like:
        a b 23.0
        which denotes an edge (a, b) with weight 23.0.

        Parameters
        ----------
        weighted : boolean
            if true, add an weighted edge
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
        self._adj[v][u] = datadict

    def remove_node(self, node_to_remove):
        try:
            neighbors = list(self._adj[node_to_remove])
            del self._node[node_to_remove]
        except KeyError:  # Node not exists in self
            raise KeyError("No node {} in graph.".format(node_to_remove))
        for neighbor in neighbors:  # Remove edges with other nodes
            del self._adj[neighbor][node_to_remove]
        del self._adj[node_to_remove]  # Remove this node

    def remove_nodes(self, nodes_to_remove: list):
        for node in nodes_to_remove:  # If not all nodes included in graph, give up removing other nodes
            assert (node in self._node), "Remove Error: No node {} in graph".format(
                node)
        for node in nodes_to_remove:
            self.remove_node(node)

    def remove_edge(self, u, v):
        try:
            del self._adj[u][v]
            if u != v:  # self-loop needs only one entry removed
                del self._adj[v][u]
        except KeyError:
            raise KeyError("No edge {}-{} in graph.".format(u, v))

    def remove_edges(self, edges_to_remove: [tuple]):
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
        return len(self._node)

    def number_of_edges(self):
        return int(self.size())

    def is_directed(self):
        return False

    def copy(self):
        G = self.__class__()
        G.graph.update(self.graph)
        for node, node_attr in self._node.items():
            G.add_node(node, **node_attr)
        for u, nbrs in self._adj.items():
            for v, edge_data in nbrs.items():
                G.add_edge(u, v, **edge_data)
        
        return G

    def nodes_subgraph(self, from_nodes: list):
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
        neighbors_of_center = list(self.all_neighbors(center))
        neighbors_of_center.append(center)
        return self.nodes_subgraph(from_nodes=neighbors_of_center)

    def to_index_node_graph(self):
        """
        Returns
        1. deepcopy of graph, with each node switched to its index.
        2. index of node
        3. node of index
        """
        G = self.__class__()
        G.graph.update(self.graph)
        index_of_node = dict()
        node_of_index = dict()
        for index, (node, node_attr) in enumerate(self._node.items()):
            G.add_node(index, **node_attr)
            index_of_node[node] = index
            node_of_index[index] = node
        for u, nbrs in self._adj.items():
            for v, edge_data in nbrs.items():
                G.add_edge(index_of_node[u], index_of_node[v], **edge_data) 
        
        return G, index_of_node, node_of_index
