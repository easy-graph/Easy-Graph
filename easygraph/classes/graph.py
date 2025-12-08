import copy
import warnings

from copy import deepcopy
from typing import Dict
from typing import List
from typing import Tuple

import easygraph as eg
import easygraph.convert as convert

from easygraph.utils.exception import EasyGraphError
from easygraph.utils.sparse import sparse_dropout


class Graph:
    """
    Base class for undirected graphs.

        Nodes are allowed for any hashable Python objects, including int, string, dict, etc.
        Edges are stored as Python dict type, with optional key/value attributes.

    Parameters
    ----------
    graph_attr : keywords arguments, optional (default : None)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    DiGraph

    Examples
    --------
    Create an empty undirected graph with no nodes and edges.

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

    gnn_data_dict_factory = dict
    raw_selfloop_dict = dict
    graph_attr_dict_factory = dict
    node_dict_factory = dict
    node_attr_dict_factory = dict
    adjlist_outer_dict_factory = dict
    adjlist_inner_dict_factory = dict
    edge_attr_dict_factory = dict
    node_index_dict = dict

    def __init__(self, incoming_graph_data=None, extra_selfloop=False, **graph_attr):
        self.graph = self.graph_attr_dict_factory()
        self._node = self.node_dict_factory()
        self._adj = self.adjlist_outer_dict_factory()
        self._raw_selfloop_dict = self.raw_selfloop_dict()
        self.extra_selfloop = extra_selfloop
        self._ndata = self.gnn_data_dict_factory()
        self.cache = {}
        self._node_index = self.node_index_dict()
        self.cflag = 0
        self._id = 0
        self.device = "cpu"
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
    def ndata(self):
        return self._ndata

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
    def node_index(self):
        return self._node_index

    @property
    def edges(self):
        """
        Return an edge list
        """
        if self.cache.get("edges") != None:
            return self.cache["edges"]
        edge_lst = list()
        seen = set()
        for u in self._adj:
            for v in self._adj[u]:
                if (u, v) not in seen:
                    seen.add((u, v))
                    seen.add((v, u))
                    edge_lst.append((u, v, self._adj[u][v]))
        del seen
        self.cache["edge"] = edge_lst
        return self.cache["edge"]

    @property
    def name(self):
        """String identifier of the graph.

        This graph attribute appears in the attribute dict G.graph
        keyed by the string `"name"`. as well as an attribute (technically
        a property) `G.name`. This is entirely user controlled.
        """
        return self.graph.get("name", "")

    @property
    def e_both_side(self, weight="weight") -> Tuple[List[List], List[float]]:
        r"""Return the list of edges including both directions."""
        if self.cache.get("e_both_side") != None:
            return self.cache["e_both_side"]
        edges = list()
        weights = list()
        seen = set()
        for u in self._adj:
            for v in self._adj[u]:
                if (u, v) not in seen:
                    seen.add((u, v))
                    seen.add((v, u))
                    edges.append([u, v])
                    edges.append([v, u])
                    if weight not in self._adj[u][v]:
                        warnings.warn("There is no property %s,default to 1" % (weight))
                        weights.append(1.0)
                        weights.append(1.0)
                    else:
                        weights.append(self._adj[u][v][weight])
                        weights.append(self._adj[v][u][weight])
        self.cache["e_both_side"] = (edges, weights)
        return self.cache["e_both_side"]

    @property
    def A(self):
        r"""Return the adjacency matrix :math:`\mathbf{A}` of the sample graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        import torch

        if self.cache.get("A", None) is None:
            if len(self.edges) == 0:
                self.cache["A"] = torch.sparse_coo_tensor(
                    size=(len(self.nodes), len(self.nodes)), device=self.device
                )
            else:
                if self.cache.get("e_both_side") is not None:
                    e_list, e_weight = self.cache["e_both_side"]

                else:
                    e_list, e_weight = self.e_both_side

                node_size = len(self.nodes)
                self.cache["A"] = torch.sparse_coo_tensor(
                    indices=torch.tensor(e_list, dtype=torch.int).t(),
                    values=torch.tensor(e_weight),
                    size=(node_size, node_size),
                    device=self.device,
                ).coalesce()
        return self.cache["A"]

    @property
    def D_v_neg_1_2(
        self,
    ):
        r"""Return the normalized diagonal matrix of vertex degree :math:`\mathbf{D}_v^{-\frac{1}{2}}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        import torch

        if self.cache.get("D_v_neg_1_2") is None:
            if self.cache.get("D_v_value") is None:
                self.cache["D_v_value"] = (
                    torch.sparse.sum(self.A, dim=1).to_dense().view(-1)
                )
                # self.cache["D_v_value"] = torch.tensor(list(self.degree().values())).float()

            _mat = self.cache["D_v_value"]
            # _mat = _tmp
            _val = _mat**-0.5
            _val[torch.isinf(_val)] = 0
            nodes_num = len(self.nodes)
            self.cache["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                torch.arange(0, len(self.nodes)).view(1, -1).repeat(2, 1),
                _val,
                torch.Size([nodes_num, nodes_num]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v_neg_1_2"]

    @property
    def index2node(self):
        """
        Assign an integer index for each node (start from 0)
        """
        if self.cache.get("index2node", None) is None:
            index2node_dict = {}
            index = 0
            # for index in range(0, len(self.nodes)):

            for index, n in enumerate(self.nodes):
                index2node_dict[index] = n
                # index += 1
            self.cache["index2node"] = index2node_dict
        return self.cache["index2node"]

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
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return the edge list, weight list and property list in the graph."""

        if self.cache.get("e", None) is None:
            node_index = self.node_index
            e_list = [
                (node_index[src_idx], node_index[dst_idx])
                for src_idx, dst_idx, d in self.edges
            ]
            w_list = []
            e_property_list = []
            v_property_list = []

            node_size = len(self.nodes)
            for i in range(0, node_size):
                v_property_list.append(self.nodes[self.index2node[i]])

            for d in self.edges:
                if "weight" not in d[2]:
                    w_list.append(1.0)
                    e_property_list.append(d[2])
                else:
                    w_list.append(d[2]["weight"])
                    tmp_dict = copy.deepcopy(d[2])
                    del tmp_dict["weight"]
                    e_property_list.append(tmp_dict)

            self.cache["e"] = e_list, w_list, v_property_list, e_property_list
        return self.cache["e"]

    @property
    def D_v(self):
        r"""Return the diagonal matrix of vertex degree :math:`\mathbf{D}_v` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        import torch

        if self.cache.get("D_v") is None:
            # print("self.A:",self.A)
            _tmp = torch.sparse.sum(self.A, dim=1).to_dense().clone().view(-1)

            nodes_num = len(self.nodes)
            self.cache["D_v"] = torch.sparse_csr_tensor(
                torch.arange(0, nodes_num + 1),
                torch.arange(0, nodes_num),
                _tmp,
                torch.Size([nodes_num, nodes_num]),
                device=self.device,
            )

            # self.cache["D_v"] = torch.sparse_coo_tensor(
            #     torch.arange(0, len(self.nodes)).view(1, -1).repeat(2, 1),
            #     _tmp,
            #     torch.Size([len(self.nodes), len(self.nodes)]),
            #     device=self.device,
            # ).coalesce()
        return self.cache["D_v"]

    def add_extra_selfloop(self):
        r"""Add extra selfloops to the graph."""
        self._has_extra_selfloop = True
        self._clear_cache()

    def remove_extra_selfloop(self):
        r"""Remove extra selfloops from the graph."""
        self._has_extra_selfloop = False
        self._clear_cache()

    def remove_selfloop(self):
        r"""Remove all selfloops from the graph."""
        self._raw_selfloop_dict.clear()
        self.remove_extra_selfloop()
        self._clear_cache()

    def nbr_v(self, v_idx: int) -> Tuple[List[int], List[float]]:
        r"""Return a vertex list of the neighbors of the vertex ``v_idx``.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        return self.N_v(v_idx).cpu().numpy().tolist()

    def N_v(self, v_idx: int) -> Tuple[List[int], List[float]]:
        import torch

        r"""Return the neighbors of the vertex ``v_idx`` with ``torch.Tensor`` format.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        sub_v_set = self.A[v_idx]._indices()[0].clone()
        return sub_v_set

    def clone(self):
        r"""Clone the graph."""
        # _g = Graph(self.num_v, extra_selfloop=self._has_extra_selfloop, device=self.device)
        # _g=self.__class__()
        # _g.device="cpu"
        # _g.extra_selfloop=False
        # _g.edges = deepcopy(self.edges)
        # _g.cache = deepcopy(self.cache)
        return self.copy()

    @name.setter
    def name(self, s):
        """
        Set graph name

        Parameters
        ----------
        s : name
        """
        self.graph["name"] = s

    def degree(self, weight="weight"):
        """Returns the weighted degree of of each node.

        Parameters
        ----------
        weight : string, optional (default: 'weight')
            Weight key of the original weighted graph.

        Returns
        -------
        degree : dict
            Each node's (key) weighted degree (value).

        Notes
        -----
        If the graph is not weighted, all the weights will be regarded as 1.

        Examples
        --------
        You can call with no attributes, if 'weight' is the weight key:

        >>> G.degree()

        if you have customized weight key 'weight_1'.

        >>> G.degree(weight='weight_1')

        """
        if self.cache.get("degree") != None:
            return self.cache["degree"]
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
        self.cache["degree"] = degree
        return degree

    def order(self):
        """Returns the number of nodes in the graph.

        Returns
        -------
        nnodes : int
            The number of nodes in the graph.

        See Also
        --------
        number_of_nodes: identical method
        __len__: identical method

        Examples
        --------
        >>> G = eg.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.order()
        3
        """
        return len(self._node)

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
        if self.cache.get("size") != None:
            return self.cache["size"]
        s = sum(d for v, d in self.degree(weight=weight).items())
        self.cache["size"] = s // 2 if weight is None else s / 2
        return self.cache["size"]

    # GCN Laplacian smoothing
    @property
    def L_GCN(self):
        r"""Return the GCN Laplacian matrix :math:`\mathcal{L}_{GCN}` of the graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.

        .. math::
            \mathcal{L}_{GCN} = \mathbf{\hat{D}}_v^{-\frac{1}{2}} \mathbf{\hat{A}} \mathbf{\hat{D}}_v^{-\frac{1}{2}}

        """
        import torch

        if self.cache.get("L_GCN") is None:
            # self.add_extra_selfloop()
            self.cache["L_GCN"] = (
                self.D_v_neg_1_2.mm(self.A).mm(self.D_v_neg_1_2).coalesce()
            )
        return self.cache["L_GCN"]

    @property
    def edge_index(self):
        import torch
        if "edge_index" not in self.cache:
            edge_list = [(u, v) for u, neighbors in self._adj.items() for v in neighbors]
            self_loops = [(u, u) for u in self._adj.keys()]
            edge_list += self_loops
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
            self.cache["edge_index"] = edge_index
        return self.cache["edge_index"]

    @property
    def norm_info(self):
        import torch

        if "norm_info" not in self.cache:
            edge_index = self.edge_index
            row, col = edge_index
            deg_dict = self.degree()
            deg = torch.tensor([deg_dict[i] for i in range(len(self.nodes))], dtype=torch.float32)
            # deg = self.degree_tensor(col, len(self.nodes), dtype=torch.float32)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            self.cache["norm_info"] = (row, col, norm)
        return self.cache["norm_info"]
    
    @property
    def adj_t(self):
        from torch_sparse import SparseTensor
        if "adj_t" not in self.cache:
            row, col, norm = self.norm_info
            N = len(self.nodes)

            self.cache["adj_t"] = SparseTensor(row=row, col=col, value=norm, sparse_sizes=(len(self.nodes), len(self.nodes)))

        return self.cache["adj_t"]

    def build_adj_gp(self, nparts: int = 4):
        if "adj_gp" not in self.cache:
            import ctypes
            import numpy as np
            import torch
            import metis
            from torch_sparse import SparseTensor
        
            row, col, norm = self.norm_info
            N = len(self.nodes)
            nnz = len(row)

            adj_list = [[] for _ in range(N)]
            for u, v in zip(row, col):
                adj_list[u].append(v)

            # --- 修改后的划分预处理 ---
            idx_t = ctypes.c_int32
            xadj = (idx_t*(N+1))()      # shape: (N+1,)
            adjncy =  (idx_t*(nnz))()  # shape: (nnz,)
            adjwgt = (idx_t*nnz)()  # shape: (nnz,)
            xadj[0] = ptr = 0
            for i, adj in enumerate(adj_list):
                for j in adj:
                    adjncy[ptr] = j
                    adjwgt[ptr] = 1
                    ptr += 1
                xadj[i+1] = ptr
            
            _, parts = metis.part_graph({
                'nvtxs': idx_t(N),  # 节点数
                'ncon': idx_t(1),
                'xadj': xadj,
                'adjncy': adjncy,
                'vwgt': None, # 节点权重
                'vsize': None,  # 节点大小 默认 None
                'adjwgt': adjwgt # 边权重
            }, nparts=nparts)

            part_to_nodes = [[] for _ in range(nparts)]
            for idx, p in enumerate(parts):
                part_to_nodes[p].append(idx)

            perm = np.concatenate(part_to_nodes)

            inv_perm = np.argsort(perm)

            row2 = inv_perm[row]
            col2 = inv_perm[col]
            adj_gp = SparseTensor(
                row=torch.tensor(row2, dtype=torch.long),
                col=torch.tensor(col2, dtype=torch.long),
                value=norm,
                sparse_sizes=(N, N)
            )
            self.cache['adj_gp'] = adj_gp
            self.cache['gp_perm'] = torch.tensor(perm)
            self.cache['gp_inv_perm'] = torch.tensor(inv_perm)
        return

    def smoothing_with_GCN(self, X, drop_rate=0.0):
        r"""Return the smoothed feature matrix with GCN Laplacian matrix :math:`\mathcal{L}_{GCN}`.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in adjacency matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        import torch

        if drop_rate > 0.0:
            L_GCN = sparse_dropout(self.L_GCN, drop_rate)
        else:
            L_GCN = self.L_GCN

        return torch.sparse.mm(L_GCN, X)

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
        """Returns an iterator of a node's neighbors.

        Parameters
        ----------
        node : Hashable
            The target node.

        Returns
        -------
        neighbors : iterator
            An iterator of a node's neighbors.

        Examples
        --------
        >>> G = eg.Graph()
        >>> G.add_edges([(1,2), (2,3), (2,4)])
        >>> for neighbor in G.neighbors(node=2):
        ...     print(neighbor)

        """
        try:
            return iter(self._adj[node])
        except KeyError:
            print("No node {}".format(node))

    all_neighbors = neighbors

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
        if "node_attr" in node_attr:
            node_attr = node_attr.get("node_attr")
        self._add_one_node(node_for_adding, node_attr)
        self._clear_cache()

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
        self._clear_cache()

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
                self._node[n] = self.node_attr_dict_factory()
            self._node[n].update(newdict)
        self._clear_cache()

    def _add_one_node(self, one_node_for_adding, node_attr: dict = {}):
        node = one_node_for_adding
        assert node != None, "Nodes can not be None."
        hash(node)
        if node not in self._node:
            self._node_index[node] = self._id
            self._id += 1
            self._adj[node] = self.adjlist_inner_dict_factory()
            attr_dict = self._node[node] = self.node_attr_dict_factory()
            attr_dict.update(node_attr)
        else:  # If already exists, there is no complain and still updating the node attribute
            self._node[node].update(node_attr)
        self._clear_cache()

    def add_edge(self, u_of_edge, v_of_edge, **edge_attr):
        """Add one edge.

        Parameters
        ----------
        u_of_edge : object
            One end of this edge

        v_of_edge : object
            The other one end of this edge

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
        if "edge_attr" in edge_attr:
            edge_attr = edge_attr.get("edge_attr")
        self._add_one_edge(u_of_edge, v_of_edge, edge_attr)
        self._clear_cache()

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
        self._clear_cache()

    def add_edges(self, edges_for_adding, edges_attr: List[Dict] = []):
        """Add a list of edges.

        Parameters
        ----------
        edges_for_adding : list of 2-element tuple
            The edges for adding. Each element is a (u, v) tuple, and u, v are
            two ends of the edge.

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
        self._clear_cache()

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
                dd = {}  # doesn't need edge_attr_dict_factory
            else:
                raise EasyGraphError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
            if u not in self._node:
                if u is None:
                    raise ValueError("None cannot be a node")
                self._adj[u] = self.adjlist_inner_dict_factory()
                self._node[u] = self.node_attr_dict_factory()
            if v not in self._node:
                if v is None:
                    raise ValueError("None cannot be a node")
                self._adj[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory()
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            self._adj[u][v] = datadict
            self._adj[v][u] = datadict
        self._clear_cache()

    def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
        """Add weighted edges in `ebunch_to_add` with specified weight attr

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the list or container will be added
            to the graph. The edges must be given as 3-tuples (u, v, w)
            where w is a number.
        weight : string, optional (default= 'weight')
            The attribute name for the edge weights to be added.
        attr : keyword arguments, optional (default= no attributes)
            Edge attributes to add/update for all edges.

        See Also
        --------
        add_edge : add a single edge
        add_edges_from : add multiple edges

        Notes
        -----
        Adding the same edge twice for Graph/DiGraph simply updates
        the edge data. For MultiGraph/MultiDiGraph, duplicate edges
        are stored.

        Examples
        --------
        >>> G = eg.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
        """
        self.add_edges_from(((u, v, {weight: d}) for u, v, d in ebunch_to_add), **attr)

    def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
        """Add weighted edges in `ebunch_to_add` with specified weight attr

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the list or container will be added
            to the graph. The edges must be given as 3-tuples (u, v, w)
            where w is a number.
        weight : string, optional (default= 'weight')
            The attribute name for the edge weights to be added.
        attr : keyword arguments, optional (default= no attributes)
            Edge attributes to add/update for all edges.

        See Also
        --------
        add_edge : add a single edge
        add_edges_from : add multiple edges

        Notes
        -----
        Adding the same edge twice for Graph/DiGraph simply updates
        the edge data. For MultiGraph/MultiDiGraph, duplicate edges
        are stored.

        Examples
        --------
        >>> G = eg.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
        """
        self.add_edges_from(((u, v, {weight: d}) for u, v, d in ebunch_to_add), **attr)

    def add_edges_from_file(self, file, weighted=False):
        """Added edges from file
        For example, txt files,

        Each line is in form like:
        a b 23.0
        which denotes an edge (a, b) with weight 23.0.

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

    def remove_nodes_from(self, nodes):
        """Remove multiple nodes.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).  If a node
            in the container is not in the graph it is silently
            ignored.

        See Also
        --------
        remove_node

        Examples
        --------
        >>> G = eg.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = list(G.nodes)
        >>> e
        [0, 1, 2]
        >>> G.remove_nodes_from(e)
        >>> list(G.nodes)
        []

        """
        adj = self._adj
        for n in nodes:
            try:
                del self._node[n]
                for u in list(adj[n]):  # list handles self-loops
                    del adj[u][n]  # (allows mutation of dict in loop)
                del adj[n]
            except KeyError:
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
        if u == v:
            self.extra_selfloop = True
            self._raw_selfloop_dict[u] = datadict
            self._clear_cache()

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
        assert node_to_remove != None, "Nodes can not be None."
        try:
            neighbors = list(self._adj[node_to_remove])
            del self._node[node_to_remove]
        except KeyError:  # Node not exists in self
            raise EasyGraphError("No node {} in graph.".format(node_to_remove))
        for neighbor in neighbors:  # Remove edges with other nodes
            del self._adj[neighbor][node_to_remove]
        del self._adj[node_to_remove]  # Remove this node
        self._clear_cache()

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
        self._clear_cache()

    def remove_edge(self, u, v):
        """Remove one edge from your graph.

        Parameters
        ----------
        u : object
            One end of the edge.

        v : object
            The other end of the edge.

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
            if u != v:  # self-loop needs only one entry removed
                del self._adj[v][u]
            self._clear_cache()
        except KeyError:
            raise KeyError("No edge {}-{} in graph.".format(u, v))

    def remove_edges(self, edges_to_remove: [tuple]):
        """Remove a list of edges from your graph.

        Parameters
        ----------
        edges_to_remove : list of tuple
            The list of edges you want to remove,
            Each element is (u, v) tuple, which denote the two ends of the edge.

        See Also
        --------
        remove_edge

        Examples
        --------
        Remove the edges *('Jack', 'Mary')* and *('Mary', 'Tom')* from *G*

        >>> G.remove_edge([
        ...     ('Jack', 'Mary'),
        ...     ('Mary', 'Tom')
        ... ])

        """
        for edge in edges_to_remove:
            u, v = edge[:2]
            self.remove_edge(u, v)
        self._clear_cache()

    def has_node(self, node):
        """Returns whether a node exists

        Parameters
        ----------
        node

        Returns
        -------
        Bool : True (exist) or False (not exists)

        """
        assert node != None, "Nodes can not be None."
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
        assert u != None and v != None, "Nodes can not be None."
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
        return False

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return False

    def copy(self):
        """Return a deep copy of the graph.

        Returns
        -------
        copy : easygraph.Graph
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

    def _clear_cache(self):
        r"""Clear the cache."""
        self.cache = {}

    def to_directed_class(self):
        """Returns the class to use for empty directed copies.

        If you subclass the base classes, use this to designate
        what directed class to use for `to_directed()` copies.
        """
        return eg.DiGraph

    def to_directed(self):
        """Creates and returns a directed graph from self.

        Returns
        -------
        G : DiGraph
            A directed graph with identical name and nodes. Each undirected
            edge (u, v, data) in the original graph is replaced by two directed
            edges (u, v, data) and (v, u, data).

        Notes
        -----
        This function returns a deepcopy of the original graph, including
        all nodes, edges, and graph. As a result, it fully duplicates
        the data and references in the original graph.

        This function differs from D=DiGraph(G) which returns a
        shallow copy.

        For more details on shallow and deep copies, refer to the
        Python `copy` module: https://docs.python.org/3/library/copy.html.

        Warning: If the original graph is a subclass of `Graph` using
        custom dict-like objects for its data structure, those customizations
        will not be preserved in the `DiGraph` created by this function.

        Examples
        --------
        Converting an undirected graph to a directed graph:

        >>> G = eg.Graph()  # or MultiGraph, etc
        >>> G.add_edge(0, 1)
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]

        Creating a deep copy of an already directed graph:

        >>> G = eg.DiGraph()  # or MultiDiGraph, etc
        >>> G.add_edge(0, 1)
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1)]
        """
        graph_class = self.to_directed_class()

        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, deepcopy(data))
            for u, nbrs in self._adj.items()
            for v, data in nbrs.items()
        )
        return G

    def cpp(self):
        G = GraphC()
        G.graph.update(self.graph)
        for u, attr in self.nodes.items():
            G.add_node(u, **attr)
        for u, v, attr in self.edges:
            G.add_edge(u, v, **attr)
        G.generate_linkgraph()
        return G


try:
    import cpp_easygraph

    class GraphC(cpp_easygraph.Graph):
        cflag = 1

except ImportError:

    class GraphC:
        def __init__(self, **graph_attr):
            print(
                "Object cannot be instantiated because C extension has not been"
                " successfully compiled and installed. Please refer to"
                " https://github.com/easy-graph/Easy-Graph/blob/master/README.rst and"
                " reinstall easygraph."
            )
            raise RuntimeError
