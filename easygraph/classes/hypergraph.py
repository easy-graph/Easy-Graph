import pickle
import random

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import easygraph as eg
import numpy as np
import torch

from easygraph.classes.base import BaseHypergraph
from easygraph.functions.drawing import draw_hypergraph
from easygraph.utils.exception import EasyGraphError
from easygraph.utils.sparse import sparse_dropout
from scipy.sparse import csr_array
from scipy.sparse import csr_matrix


if TYPE_CHECKING:
    from easygraph import Graph

__all__ = ["Hypergraph"]


class Hypergraph(BaseHypergraph):

    """
    The ``Hypergraph`` class is developed for hypergraph structures.
    Please notice that node id in hypergraph is in [0, num_v)

    Parameters
    ----------
        num_v  : (int) The number of vertices in the hypergraph
        e_list : (Union[List[int], List[List[int]]], optional) A list of hyperedges describes how the vertices point to the hyperedges. Defaults to ``None``
        v_property: Optional[List[Dict]], A list of node properties. Defaults to ``None``
        e_property: Optional[List[Dict]], A list of hyperedges properties. Defaults to ``None``
        e_weight : (Union[float, List[float]], optional)  A list of weights for hyperedges. If set to None, the value ``1`` is used for all hyperedges. Defaults to None
        merge_op : (str) The operation to merge those conflicting hyperedges in the same hyperedge group, which can be ``'mean'``, ``'sum'`` or ``'max'``. Defaults to ``'mean'``
        device : (torch.device, optional) The device to store the hypergraph. Defaults to torch.device('cpu')


    """

    gnn_data_dict_factory = dict
    degree_data_dict = dict

    def __init__(
        self,
        num_v: int,
        v_property: Optional[List[Dict]] = None,
        e_list: Optional[Union[List[int], List[List[int]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        e_property: Optional[List[Dict]] = None,
        merge_op: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            num_v,
            e_list=e_list,
            v_property=v_property,
            e_property=e_property,
            device=device,
        )

        self._ndata = self.gnn_data_dict_factory()
        self.deg_v_dict = self.degree_data_dict()
        self.n_e_dict = {}
        self.edge_index = -1
        self.device = device
        for i in range(num_v):
            self.deg_v_dict[i] = 0
            self.n_e_dict[i] = []

        if e_list is not None:
            self.add_hyperedges(
                e_list=e_list,
                e_weight=e_weight,
                e_property=e_property,
                merge_op=merge_op,
            )
        edges_col = []
        indptr_list = []
        ptr = 0
        for v in self.n_e_dict.values():
            edges_col.extend(v)
            indptr_list.append(ptr)
            ptr += len(v)
        indptr_list.append(ptr)
        e_idx, v_idx = [], []
        for n, e in self.n_e_dict.items():
            v_idx.extend([n] * len(e))
            e_idx.extend(e)
        self.cache["e_idx"] = e_idx
        self.cache["v_idx"] = v_idx

        self.cache["edges_col"] = np.array(edges_col)
        self.cache["indptr_list"] = np.array(indptr_list)

    def __repr__(self) -> str:
        r"""Print the hypergraph information."""
        return f"Hypergraph(num_vertex={self.num_v}, num_hyperedge={self.num_e})"

    @property
    def ndata(self):
        return self._ndata

    @property
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the hypergraph."""
        return {
            "num_v": self.num_v,
            "v_property": self.v_property,
            "e_property": self.e_property,
            "raw_groups": self._raw_groups,
            "deg_v_dict": self.deg_v_dict,
        }

    def unique_edge_sizes(self):
        """A function that returns the unique edge sizes.

        Returns
        -------
        list()
            The unique edge sizes in ascending order by size.
        """
        edge_size_set = set()
        edge_lst = self.e[0]
        for e in edge_lst:
            edge_size_set.add(len(e))

        return sorted(edge_size_set)

    def is_uniform(self):
        """Order of uniformity if the hypergraph is uniform, or False.

        A hypergraph is uniform if all its edges have the same order.

        Returns d if the hypergraph is d-uniform, that is if all edges
        in the hypergraph (excluding singletons) have the same degree d.
        Returns False if not uniform.

        Returns
        -------
        d : int or False
            If the hypergraph is d-uniform, return d, or False otherwise.

        Examples
        --------
        This function can be used as a boolean check:

        >>> import easygraph as eg
        >>> H = eg.Hypergraph(v_num = 5, e_list = [(0, 1, 2), (1, 2, 3), (2, 3, 4)])
        >>> H.is_uniform()
        2
        """
        edge_sizes = self.unique_edge_sizes()
        if 1 in edge_sizes:
            edge_sizes.remove(1)

        if edge_sizes is None or len(edge_sizes) != 1:
            return False

        # order of all edges
        return edge_sizes.pop()

    def save(self, file_path: Union[str, Path]):
        r"""Save the EasyGraph's hypergraph structure a file.

        Parameters:
            ``file_path`` (``Union[str, Path]``): The file path to store the EasyGraph's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.parent.exists(), "The directory does not exist."
        data = {
            "class": "Hypergraph",
            "state_dict": self.state_dict,
        }
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the EasyGraph's hypergraph structure from a file.

        Parameters:
            ``file_path`` (``Union[str, Path]``): The file path to load the EasyGraph's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.exists(), "The file does not exist."
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        assert (
            data["class"] == "Hypergraph"
        ), "The file is not a EasyGraph's hypergraph file."
        return Hypergraph.from_state_dict(data["state_dict"])

    def draw(
        self,
        e_style: str = "circle",
        v_label: Optional[List[str]] = None,
        v_size: Union[float, list] = 1.0,
        v_color: Union[str, list] = "r",
        v_line_width: Union[str, list] = 1.0,
        e_color: Union[str, list] = "gray",
        e_fill_color: Union[str, list] = "whitesmoke",
        e_line_width: Union[str, list] = 1.0,
        font_size: float = 1.0,
        font_family: str = "sans-serif",
        push_v_strength: float = 1.0,
        push_e_strength: float = 1.0,
        pull_e_strength: float = 1.0,
        pull_center_strength: float = 1.0,
    ):
        r"""Draw the hypergraph structure.

        Parameters:
            ``e_style`` (``str``): The style of hyperedges. The available styles are only ``'circle'``. Defaults to ``'circle'``.
            ``v_label`` (``list``): The labels of vertices. Defaults to ``None``.
            ``v_size`` (``float`` or ``list``): The size of vertices. Defaults to ``1.0``.
            ``v_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices. Defaults to ``'r'``.
            ``v_line_width`` (``float`` or ``list``): The line width of vertices. Defaults to ``1.0``.
            ``e_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'gray'``.
            ``e_fill_color`` (``str`` or ``list``): The fill `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'whitesmoke'``.
            ``e_line_width`` (``float`` or ``list``): The line width of hyperedges. Defaults to ``1.0``.
            ``font_size`` (``float``): The font size of labels. Defaults to ``1.0``.
            ``font_family`` (``str``): The font family of labels. Defaults to ``'sans-serif'``.
            ``push_v_strength`` (``float``): The strength of pushing vertices. Defaults to ``1.0``.
            ``push_e_strength`` (``float``): The strength of pushing hyperedges. Defaults to ``1.0``.
            ``pull_e_strength`` (``float``): The strength of pulling hyperedges. Defaults to ``1.0``.
            ``pull_center_strength`` (``float``): The strength of pulling vertices to the center. Defaults to ``1.0``.
        """
        draw_hypergraph(
            self,
            e_style,
            v_label,
            v_size,
            v_color,
            v_line_width,
            e_color,
            e_fill_color,
            e_line_width,
            font_size,
            font_family,
            push_v_strength,
            push_e_strength,
            pull_e_strength,
            pull_center_strength,
        )

    def clear(self):
        r"""Clear all hyperedges and caches from the hypergraph."""

        super().clear()
        self.deg_v_dict = {}
        self._ndata = {}

    def clone(self) -> "Hypergraph":
        r"""Return a copy of the hypergraph."""
        hg = Hypergraph(self.num_v, device=self.device)
        hg._raw_groups = deepcopy(self._raw_groups)
        hg.cache = deepcopy(self.cache)
        hg.group_cache = deepcopy(self.group_cache)
        hg.deg_v_dict = deepcopy(self.deg_v_dict)
        return hg

    def to(self, device: torch.device):
        r"""Move the hypergraph to the specified device.

        Parameters:
            ``device`` (``torch.device``): The target device.
        """
        return super().to(device)

    # =====================================================================================
    # some construction functions
    @staticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the hypergraph from the state dict.

        Parameters:
            ``state_dict`` (``dict``): The state dict to load the hypergraph.
        """
        _hg = Hypergraph(state_dict["num_v"])
        _hg._raw_groups = deepcopy(state_dict["raw_groups"])
        _hg._e_property = deepcopy(state_dict["e_property"])
        _hg._v_property = deepcopy(state_dict["v_property"])
        _hg.deg_v_dict = deepcopy(state_dict["deg_v_dict"])
        return _hg

    @staticmethod
    def _e_list_from_feature_kNN(features: torch.Tensor, k: int):
        import scipy

        r"""Construct hyperedges from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k-1` neighbor vertices.

        Parameters:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
        """
        features = features.cpu().numpy()
        assert features.ndim == 2, "The feature matrix should be 2-D."
        assert k <= features.shape[0], (
            "The number of nearest neighbors should be less than or equal to the number"
            " of vertices."
        )
        tree = scipy.spatial.cKDTree(features)
        _, nbr_array = tree.query(features, k=k)
        return nbr_array.tolist()

    @staticmethod
    def from_feature_kNN(
        features: torch.Tensor, k: int, device: torch.device = torch.device("cpu")
    ):
        r"""Construct the hypergraph from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k-1` neighbor vertices.

        .. note::
            The constructed hypergraph is a k-uniform hypergraph. If the feature matrix has the size :math:`N \times C`, the number of vertices and hyperedges of the constructed hypergraph are both :math:`N`.

        Parameters:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_feature_kNN(features, k)
        hg = Hypergraph(num_v=features.shape[0], e_list=e_list, device=device)
        return hg

    @staticmethod
    def from_graph(graph, device: torch.device = torch.device("cpu")) -> "Hypergraph":
        r"""Construct the hypergraph from the graph. Each edge in the graph is treated as a hyperedge in the constructed hypergraph.

        .. note::
            The constructed hypergraph is a 2-uniform hypergraph, and has the same number of vertices and edges/hyperedges as the graph.

        Parameters:
            ``graph`` (``eg.Graph``): The graph to construct the hypergraph.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list, e_weight, v_property, e_property = graph.e
        hg = Hypergraph(
            num_v=len(graph.nodes),
            e_list=e_list,
            e_weight=e_weight,
            v_property=v_property,
            e_property=e_property,
            device=device,
        )
        return hg

    @staticmethod
    def _e_list_from_graph_kHop(
        graph,
        k: int,
        only_kHop: bool = False,
    ) -> List[tuple]:
        r"""Construct the hyperedge list from the graph by k-Hop neighbors. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Parameters:
            ``graph`` (``eg.Graph``): The graph to construct the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``, optional): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
        """
        assert (
            k >= 1
        ), "The number of hop neighbors should be larger than or equal to 1."
        A_1, A_k = graph.A.clone(), graph.A.clone()
        A_history = []
        for _ in range(k - 1):
            A_k = torch.sparse.mm(A_k, A_1)
            if not only_kHop:
                A_history.append(A_k.clone())
        if not only_kHop:
            A_k = A_1
            for A_ in A_history:
                A_k = A_k + A_
        e_list = [
            tuple(set([v_idx] + A_k[v_idx]._indices().cpu().squeeze(0).tolist()))
            for v_idx in range(len(graph.nodes))
        ]
        return e_list

    @staticmethod
    def from_graph_kHop(
        graph,
        k: int,
        only_kHop: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> "Hypergraph":
        r"""Construct the hypergraph from the graph by k-Hop neighbors. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Parameters:
            ``graph`` (``eg.Graph``): The graph to construct the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_graph_kHop(graph, k, only_kHop)
        hg = Hypergraph(num_v=len(graph.nodes), e_list=e_list, device=device)
        return hg

    def isOutRange(self, id):
        if id >= self.num_v or id < 0:
            return False
        return True

    def add_hyperedges(
        self,
        e_list: Union[List[int], List[List[int]]],
        e_weight: Optional[Union[float, List[float]]] = None,
        e_property: Optional[Union[Dict, List[Dict]]] = None,
        merge_op: str = "sum",
        group_name: str = "main",
    ):
        r"""Add hyperedges to the hypergraph. If the ``group_name`` is not specified, the hyperedges will be added to the default ``main`` hyperedge group.

        Parameters:
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges. The possible values are ``"mean"``, ``"sum"``, and ``"max"``. Defaults to ``"mean"``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        e_list = self._format_e_list(e_list)
        if e_weight is None:
            e_weight = [1.0] * len(e_list)
        elif type(e_weight) in (int, float):
            e_weight = [e_weight]
        elif type(e_weight) is list:
            pass
        else:
            raise TypeError(
                "The type of e_weight should be float or list, but got"
                f" {type(e_weight)}"
            )
        assert len(e_list) == len(
            e_weight
        ), "The number of hyperedges and the number of weights are not equal."

        for _idx in range(len(e_list)):
            flag = True
            if (
                group_name not in self._raw_groups
                or self._hyperedge_code(e_list[_idx], e_list[_idx])
                not in self._raw_groups[group_name]
            ):
                flag = False
                self.edge_index += 1
            for n_id in e_list[_idx]:
                if self.isOutRange(n_id) == False:
                    raise EasyGraphError(
                        "The node id:"
                        + str(n_id)
                        + " in hyperedge is out of range, please ensure that"
                        " the node is in [0,n)"
                    )
                self.deg_v_dict[n_id] += 1
                if flag is False:
                    self.n_e_dict[n_id].append(self.edge_index)

            if e_property != None:
                if type(e_property) == dict:
                    e_property = [e_property]
                e_property[_idx].update({"w_e": float(e_weight[_idx])})
                self._add_hyperedge(
                    self._hyperedge_code(e_list[_idx], e_list[_idx]),
                    e_property[_idx],
                    merge_op,
                    group_name,
                )
            else:
                self._add_hyperedge(
                    self._hyperedge_code(e_list[_idx], e_list[_idx]),
                    {"w_e": float(e_weight[_idx])},
                    merge_op,
                    group_name,
                )

        self._clear_cache(group_name)

    def add_hyperedges_from_feature_kNN(
        self, feature: torch.Tensor, k: int, group_name: str = "main"
    ):
        r"""Add hyperedges from the feature matrix by k-NN. Each hyperedge is constructed by the central vertex and its :math:`k`-Nearest Neighbor vertices.

        Parameters:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert feature.shape[0] == self.num_v, (
            "The number of vertices in the feature matrix is not equal to the number of"
            " vertices in the hypergraph."
        )
        e_list = Hypergraph._e_list_from_feature_kNN(feature, k)
        self.add_hyperedges(e_list, group_name=group_name)

    def add_hyperedges_from_graph(self, graph, group_name: str = "main"):
        r"""Add hyperedges from edges in the graph. Each edge in the graph is treated as a hyperedge.

        Parameters:
            ``graph`` (``eg.Graph``): The graph to join the hypergraph.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == len(
            graph.nodes
        ), "The number of vertices in the hypergraph and the graph are not equal."
        e_list, e_weight = graph.e_both_side
        self.add_hyperedges(e_list, e_weight=e_weight, group_name=group_name)

    def add_hyperedges_from_graph_kHop(
        self, graph, k: int, only_kHop: bool = False, group_name: str = "main"
    ):
        r"""Add hyperedges from vertices and its k-Hop neighbors in the graph. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Parameters:
            ``graph`` (``eg.Graph``): The graph to join the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == len(
            graph.nodes
        ), "The number of vertices in the hypergraph and the graph are not equal."
        e_list = Hypergraph._e_list_from_graph_kHop(graph, k, only_kHop=only_kHop)
        self.add_hyperedges(e_list, group_name=group_name)

    def remove_hyperedges(
        self,
        e_list: Union[List[int], List[List[int]]],
        group_name: Optional[str] = None,
    ):
        r"""Remove the specified hyperedges from the hypergraph.

        Parameters:
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``group_name`` (``str``, optional): Remove these hyperedges from the specified hyperedge group. If not specified, the function will
                remove those hyperedges from all hyperedge groups. Defaults to the ``None``.
        """
        assert (
            group_name is None or group_name in self.group_names
        ), "The specified group_name is not in existing hyperedge groups."
        e_list = self._format_e_list(e_list)
        if group_name is None:
            for _idx in range(len(e_list)):
                for n_id in e_list[_idx]:
                    self.deg_v_dict[n_id] -= 1
                    if self.isOutRange(n_id) == False:
                        raise EasyGraphError(
                            "The node id in hyperedge is out of range, please ensure"
                            " that the node is in [1,n)"
                        )
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                for name in self.group_names:
                    self._raw_groups[name].pop(e_code, None)
        else:
            for _idx in range(len(e_list)):
                for n_id in e_list[_idx]:
                    self.deg_v_dict[n_id] -= 1
                    if self.isOutRange(n_id) == False:
                        raise EasyGraphError(
                            "The node id in hyperedge is out of range, please ensure"
                            " that the node is in [1,n)"
                        )
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                self._raw_groups[group_name].pop(e_code, None)

        self.edge_index = -1
        self.n_e_dict = {i: [] for i in range(self.num_v)}
        for e in self.e[0]:
            self.edge_index += 1
            for n_id in e:
                self.n_e_dict[n_id].append(self.edge_index)
        self._clear_cache(group_name)

    def remove_group(self, group_name: str):
        r"""Remove the specified hyperedge group from the hypergraph.

        Parameters:
            ``group_name`` (``str``): The name of the hyperedge group to remove.
        """
        for e_code, e in self._raw_groups[group_name].items():
            e = e_code[0]
            for n_id in e:
                self.deg_v_dict[n_id] -= 1
        self._raw_groups.pop(group_name, None)
        self._clear_cache(group_name)

    def drop_hyperedges(self, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the hypergraph. This function will return a new hypergraph with non-dropped hyperedges.

        Parameters:
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                _raw_groups[name] = {
                    k: v
                    for k, v in self._raw_groups[name].items()
                    if random.random() > drop_rate
                }
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
                "e_property": self._e_property,
                "v_property": self._v_property,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unknown drop order: {ord}.")
        return _hg

    def drop_hyperedges_of_group(
        self, group_name: str, drop_rate: float, ord="uniform"
    ):
        r"""Randomly drop hyperedges from the specified hyperedge group. This function will return a new hypergraph with non-dropped hyperedges.

        Parameters:
            ``group_name`` (``str``): The name of the hyperedge group.
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                if name == group_name:
                    _raw_groups[name] = {
                        k: v
                        for k, v in self._raw_groups[name].items()
                        if random.random() > drop_rate
                    }
                else:
                    _raw_groups[name] = self._raw_groups[name]
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": self._raw_groups,
                "e_property": self._e_property,
                "v_property": self._v_property,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unknown drop order: {ord}.")
        return _hg

    # =====================================================================================
    # properties for representation
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices."""
        return super().v

    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the hypergraph."""
        if self.cache.get("e", None) is None:
            e_list, e_weight, e_property = [], [], []
            for name in self.group_names:
                _e = self.e_of_group(name)
                e_list.extend(_e[0])
                e_weight.extend(_e[1])
                e_property.extend(_e[2])
            self.cache["e"] = (e_list, e_weight, e_property)
        return self.cache["e"]

    def e_of_group(self, group_name: str) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights of the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("e", None) is None:
            e_list = [e_code[0] for e_code in self._raw_groups[group_name].keys()]
            e_weight = [
                e_content["w_e"] for e_content in self._raw_groups[group_name].values()
            ]

            e_property = []
            for e_content in self._raw_groups[group_name].values():
                properties = {}
                for k, v in e_content.items():
                    if k != "w_e":
                        properties[k] = v
                e_property.append(properties)
            self.group_cache[group_name]["e"] = (e_list, e_weight, e_property)
        return self.group_cache[group_name]["e"]

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the hypergraph."""
        return super().num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of hyperedges in the hypergraph."""
        return super().num_e

    def num_e_of_group(self, group_name: str) -> int:
        r"""Return the number of hyperedges of the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        return super().num_e_of_group(group_name)

    @property
    def deg_v(self) -> List[int]:
        r"""Return the degree list of each vertex."""
        return self.D_v.to_sparse_coo()._values().cpu().view(-1).numpy().tolist()

    def deg_v_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each vertex of the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_v_of_group(group_name)._values().cpu().view(-1).numpy().tolist()

    @property
    def deg_e(self) -> List[int]:
        r"""Return the degree list of each hyperedge."""
        return self.D_e.to_sparse_coo()._values().cpu().view(-1).numpy().tolist()

    def deg_e_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each hyperedge of the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_e_of_group(group_name)._values().cpu().view(-1).numpy().tolist()

    def nbr_e(self, v_idx: int) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex.

        Parameters:
            ``v_idx`` (``int``): The index of the vertex.
        """
        return self.N_e(v_idx).cpu().numpy().tolist()

    def nbr_e_of_group(self, v_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex of the specified hyperedge group.

        Parameters:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_e_of_group(v_idx, group_name).cpu().numpy().tolist()

    def nbr_v(self, e_idx: int) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge.

        Parameters:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        return self.N_v(e_idx).cpu().numpy().tolist()

    def nbr_v_of_group(self, e_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge of the specified hyperedge group.

        Parameters:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_v_of_group(e_idx, group_name).cpu().numpy().tolist()

    @property
    def num_groups(self) -> int:
        r"""Return the number of hyperedge groups in the hypergraph."""
        return super().num_groups

    @property
    def group_names(self) -> List[str]:
        r"""Return the names of all hyperedge groups in the hypergraph."""
        return super().group_names

    # =====================================================================================
    # properties for deep learning
    @property
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in the hypergraph including

        Sparse Matrices:

        .. math::
            \mathbf{H}, \mathbf{H}^\top, \mathcal{L}_{sym}, \mathcal{L}_{rw} \mathcal{L}_{HGNN},

        Sparse Diagnal Matrices:

        .. math::
            \mathbf{W}_e, \mathbf{D}_v, \mathbf{D}_v^{-1}, \mathbf{D}_v^{-\frac{1}{2}}, \mathbf{D}_e, \mathbf{D}_e^{-1},

        Vectors:

        .. math::
            \overrightarrow{v2e}_{src}, \overrightarrow{v2e}_{dst}, \overrightarrow{v2e}_{weight},\\
            \overrightarrow{e2v}_{src}, \overrightarrow{e2v}_{dst}, \overrightarrow{e2v}_{weight}

        """
        return [
            "H",
            "H_T",
            "L_sym",
            "L_rw",
            "L_HGNN",
            "W_e",
            "D_v",
            "D_v_neg_1",
            "D_v_neg_1_2",
            "D_e",
            "D_e_neg_1",
            "v2e_src",
            "v2e_dst",
            "v2e_weighte2v_src",
            "e2v_dst",
            "e2v_weight",
        ]

    @property
    def v2e_src(self) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._indices()[1].clone()

    def v2e_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[1].clone()

    @property
    def v2e_dst(self) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        if self.cache.get("v2e_dst") is None:
            self.cache["v2e_dst"] = self.H_T._indices()[0]
        return self.cache["v2e_dst"]

    def v2e_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[0].clone()

    @property
    def v2e_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._values().clone()

    def v2e_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._values().clone()

    @property
    def e2v_src(self) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[1]

    def e2v_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[1].clone()

    @property
    def e2v_dst(self) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[0].clone()

    def e2v_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[0].clone()

    @property
    def e2v_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._values()

    def e2v_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._values().clone()

    @property
    def H(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` with ``torch.Tensor`` format.
        """

        if self.cache.get("H") is None:
            num_e = len(self._raw_groups["main"])
            if self.cache.get("v_idx") is None or self.cache.get("e_idx") is None:
                e_idx, v_idx = [], []
                for n, e in self.n_e_dict.items():
                    v_idx.extend([n] * len(e))
                    e_idx.extend(e)
                self.cache["e_idx"] = e_idx
                self.cache["v_idx"] = v_idx
            self.cache["H"] = torch.sparse_coo_tensor(
                torch.tensor(
                    [self.cache["v_idx"], self.cache["e_idx"]], dtype=torch.long
                ),
                torch.ones(len(self.cache["v_idx"])),
                torch.Size([self.num_v, num_e]),
            ).coalesce()

        return self.cache["H"]

    @property
    def e_set(self):
        if self.cache.get("e_set") is None:
            e_lst = []
            for name in self.group_names:
                _e = self.e_of_group(name)
                e_lst.extend(_e[0])
            self.cache["e_set"] = e_lst
        return self.cache["e_set"]

    @property
    def incidence_matrix(self):
        if self.cache.get("incidence_matrix") is None:
            if (
                self.cache.get("edges_col") is None
                or self.cache.get("indptr_list") is None
            ):
                edges_col = []
                indptr_list = []
                ptr = 0
                for v in self.n_e_dict.values():
                    edges_col.extend(v)
                    indptr_list.append(ptr)
                    ptr += len(v)
                indptr_list.append(ptr)
                self.cache["edges_col"] = np.array(edges_col)
                self.cache["indptr_list"] = np.array(indptr_list)
            H = csr_matrix(
                (
                    [1] * len(self.cache["edges_col"]),
                    self.cache["edges_col"],
                    self.cache["indptr_list"],
                ),
                shape=(self.num_v, self.num_e),
                dtype=int,
            )
            self.cache["incidence_matrix"] = H
        return self.cache["incidence_matrix"]

    def get_star_expansion(self):
        r"""
        The star expansion algorithm creates a graph  G*(V*, E*) for every hypergraph G(V, E).
        The graph G*(V*, E*) introduces a node e∈E for each hyperedge in G(V, E), where V* = V ∪ E.
        Each node e is connected to all the nodes belonging to the hyperedge it originates from, i.e., E* = {(u, e): u∈e, e∈E}.
        It is worth noting that each hyperedge in the set E corresponds to a star-shaped structure in the graph G*(V*, E*),
        and G* is a bipartite graph. The star expansion redistributes the weights of hyperedges to their corresponding ordinary pairwise graph edges.

        $ \omega ^{*}(u,e)=\frac{\omega(e)}{\delta(e)} $

        References
        ----------
        Antelmi, Alessia, et al. "A survey on hypergraph representation learning." ACM Computing Surveys 56.1 (2023): 1-38.

        """
        star_expansion_graph = eg.Graph()
        for node in self.v:
            star_expansion_graph.add_node(node, type="node")
        e_index = len(self.v)
        hyperedge_edge_list = self.e[0]
        hyperedge_weight_list = self.e[1]
        hyperedge_property_list = self.e[2]
        for hyperedge_index, e in enumerate(hyperedge_edge_list):
            hyperedge_weight = hyperedge_weight_list[hyperedge_index]
            star_expansion_graph.add_node(e_index, type="hyperedge")
            for index, node in enumerate(e):
                star_expansion_graph.add_edge(
                    e_index,
                    node,
                    weight=hyperedge_weight / len(e),
                    hyperedge_index=hyperedge_index,
                    edge_property=hyperedge_property_list[index],
                )
            e_index = e_index + 1
        return star_expansion_graph

    def neighbor_of_node(self, node):
        neighbor_lst = list()
        node_adj = self.adjacency_matrix()
        if (
            self.cache.get("neighbor") is None
            or self.cache["neighbor"].get(node) is None
        ):
            start = node_adj.indptr[node]
            end = node_adj.indptr[node + 1]

            for j in range(start, end):
                neighbor_lst.append(node_adj.indices[j])

            if self.cache.get("neighbor") is None:
                self.cache["neighbor"] = {}
                self.cache["neighbor"][node] = neighbor_lst
            else:
                self.cache["neighbor"][node] = neighbor_lst

        return self.cache["neighbor"][node]

    def adjacency_matrix(self, s=1, weight=False):
        r"""
        The :term:`s-adjacency matrix` for the dual hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        Returns
        -------
        adjacency_matrix : scipy.sparse.csr.csr_matrix

        """
        if self.cache.get("adjacency_matrix") == None:
            tmp_H = self.incidence_matrix
            A = tmp_H @ tmp_H.T
            A[np.diag_indices_from(A)] = 0
            if not weight:
                A = (A >= s) * 1
            self.cache["adjacency_matrix"] = csr_matrix(A)
        return self.cache["adjacency_matrix"]

    def edge_adjacency_matrix(self, s=1, weight=False):
        r"""
        The :term:`s-adjacency matrix` for the dual hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        Returns
        -------
        adjacency_matrix : scipy.sparse.csr.csr_matrix

        """
        tmp_H = self.incidence_matrix
        A = (tmp_H.T) @ (tmp_H)
        A[np.diag_indices_from(A)] = 0
        if not weight:
            A = (A >= s) * 1
        return csr_array(A)

    def _fetch_H(self, direction="v2e", group_name="main"):
        r"""Fetch the H matrix of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.
        Args:
            ``direction`` (``str``): The direction of hyperedges can be either ``'v2e'`` or ``'e2v'``.
            ``group_name`` (``str``): The name of the group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        assert direction in ["v2e", "e2v"], "direction must be one of ['v2e', 'e2v']"
        if direction == "v2e":
            select_idx = 0
        else:
            select_idx = 1
        num_e = len(self._raw_groups[group_name])
        e_idx, v_idx = [], []
        for _e_idx, e in enumerate(self._raw_groups[group_name].keys()):
            sub_e = e[select_idx]
            v_idx.extend(sub_e)
            e_idx.extend([_e_idx] * len(sub_e))

        H = torch.sparse_coo_tensor(
            torch.tensor([v_idx, e_idx], dtype=torch.long),
            torch.ones(len(v_idx)),
            torch.Size([self.num_v, num_e]),
            device=self.device,
        ).coalesce()
        return H
        # if self.cache.get("main_H") is None:
        #     num_e = len(self._raw_groups[group_name])
        #     self.cache["main_H"] = torch.sparse_coo_tensor(
        #         ([self.cache["v_idx"], self.cache["e_idx"]]),
        #         torch.ones(len(self.cache["v_idx"])),
        #         torch.Size([self.num_v, num_e]),
        #         device=self.device,
        #     ).coalesce()
        # return self.cache["main_H"]

    def H_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` of the specified hyperedge group with ``torch.Tensor`` format.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H") is None:
            self.group_cache[group_name]["H"] = self._fetch_H()
        return self.group_cache[group_name]["H"]

    def edge_distance(self, source, target, s=1):
        """

        Parameters
        ----------
        source
        target
        s

        Returns
        -------
        s- walk distance : the shortest s-walk edge distance

        Notes
        -----
            The s-distance is the shortest s-walk length between the edges.
            An s-walk between edges is a sequence of edges such that
            consecutive pairwise edges intersect in at least s nodes. The
            length of the shortest s-walk is 1 less than the number of edges
            in the path sequence.

        """
        l_graph = self.get_clique_expansion(s=s, edge=True)
        if source not in l_graph.nodes:
            raise EasyGraphError("Please make sure source exist!")
        dist = eg.Dijkstra(l_graph, source)
        if target in dist:
            return dist[target]
        raise EasyGraphError("Please make sure target exist!")

    def distance(self, source, target=None, s=1):
        """

        Parameters
        ----------
        source : node in the hypergraph
        target : node in the hypergraph
        s : positive integer
            the number of edges

        Returns
        -------
        s-walk distance : int

        Notes
        -----
        The s-distance is the shortest s-walk length between the nodes.
        An s-walk between nodes is a sequence of nodes that pairwise share
        at least s edges. The length of the shortest s-walk is 1 less than
        the number of nodes in the path sequence.

        Uses the EasyGraph's Dijkstra method on the graph
        generated by the s-adjacency matrix.

        """

        l_graph = self.get_clique_expansion(s=s)
        if source not in l_graph.nodes:
            raise EasyGraphError("Please make sure source exist!")
        if target is not None and target not in l_graph.nodes:
            raise EasyGraphError("Please make sure target exist!")
        dist = eg.single_source_dijkstra(G=l_graph, source=source, target=target)
        return dist[target] if target != None else dist

    def edge_diameter(self, s=1):
        """
        Returns the length of the longest shortest s-walk between edges in
        hypergraph

        Parameters
        ----------
        s : int, optional, default 1

        Return
        ------
        edge_diameter : int

        Raises
        ------
        EasyGraphXError
            If hypergraph is not s-edge-connected

        Notes
        -----
        Two edges are s-adjacent if they share s nodes.
        Two nodes e_start and e_end are s-walk connected if there is a
        sequence of edges e_start, e_1, e_2, ... e_n-1, e_end such that
        consecutive edges are s-adjacent. If the graph is not connected, an
        error will be raised.

        """
        l_graph = self.get_clique_expansion(s=s, edge=True)
        if eg.is_connected(l_graph):
            return eg.diameter(l_graph)
        raise EasyGraphError(f"Hypergraph is not s-connected. s={s}")

    def diameter(self, s=1):
        """
        Returns the length of the longest shortest s-walk between nodes in
        hypergraph

        Parameters
        ----------
        s : int, optional, default 1

        Returns
        -------
        diameter : int
        Raises
        ------
        EasyGraphError
            If hypergraph is not s-edge-connected

        Notes
        -----
        Two nodes are s-adjacent if they share s edges.
        Two nodes v_start and v_end are s-walk connected if there is a
        sequence of nodes v_start, v_1, v_2, ... v_n-1, v_end such that
        consecutive nodes are s-adjacent. If the graph is not connected,
        an error will be raised.
        """
        l_graph = self.get_clique_expansion(s=s)
        if eg.is_connected(l_graph):
            return eg.diameter(l_graph)
        raise EasyGraphError(f"Hypergraph is not s-connected. s={s}")

    @property
    def H_T(self) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` with ``torch.Tensor`` format.
        """
        if self.cache.get("H_T") is None:
            self.cache["H_T"] = self.H.t()
        return self.cache["H_T"]

    def H_T_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` of the specified hyperedge group with ``torch.Tensor`` format.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_T") is None:
            self.group_cache[group_name]["H_T"] = self.H_of_group(group_name).t()
        return self.group_cache[group_name]["H_T"]

    @property
    def W_e(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges with ``torch.Tensor`` format.
        """
        if self.cache.get("W_e") is None:
            _tmp = torch.tensor(self.e[1])
            _num_e = _tmp.size(0)
            self.cache["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
            ).coalesce()

        return self.cache["W_e"]

    def W_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges of the specified hyperedge group with ``torch.Tensor`` format.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("W_e") is None:
            w_list = [1.0] * len(self._raw_groups["main"])
            _tmp = torch.tensor(w_list, device=self.device).view((-1, 1)).view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["W_e"]

    @property
    def degree_node(self):
        return self.deg_v_dict

    @property
    def D_v(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v") is None:
            if self.cache.get("D_v_value") is None:
                self.cache["D_v_value"] = (
                    torch.sparse.sum(self.H, dim=1).to_dense().view(-1)
                )

            self.cache["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, self.num_v).view(1, -1).repeat(2, 1),
                self.cache["D_v_value"],
                torch.Size([self.num_v, self.num_v]),
            ).coalesce()
        return self.cache["D_v"]

    def D_v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v") is None:
            _tmp = (
                torch.sparse.sum(self.H_of_group(group_name), dim=1)
                .to_dense()
                .clone()
                .view(-1)
            )
            _num_v = _tmp.size(0)
            self.group_cache[group_name]["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_v).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                # device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_v"]

    @property
    def D_v_neg_1(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1") is None:
            if self.cache.get("D_v_value") is None:
                self.cache["D_v_value"] = (
                    torch.sparse.sum(self.H, dim=1).to_dense().view(-1)
                )
            _tmp = self.cache["D_v_value"]
            _num_v = _tmp.size(0)
            _val = _tmp**-1
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1"] = torch.sparse_csr_tensor(
                torch.arange(0, _num_v + 1),
                torch.arange(0, _num_v),
                _val,
                torch.Size([_num_v, _num_v]),
                # device=self.device,
            )

        return self.cache["D_v_neg_1"]

    def D_v_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1"]

    @property
    def D_v_neg_1_2(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1_2") is None:
            if self.cache.get("D_v_value") is None:
                self.cache["D_v_value"] = (
                    torch.sparse.sum(self.H, dim=1).to_dense().view(-1)
                )
            _mat = self.cache["D_v_value"]
            _mat = _mat**-0.5
            _mat[torch.isinf(_mat)] = 0
            self.cache["D_v_neg_1_2"] = torch.sparse_csr_tensor(
                torch.arange(0, self.num_v + 1),
                torch.arange(0, self.num_v),
                _mat,
                torch.Size([self.num_v, self.num_v]),
            )

        return self.cache["D_v_neg_1_2"]

    def D_v_neg_1_2_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1_2") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1_2"]

    @property
    def D_e(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e") is None:
            _tmp = torch.sparse.sum(self.H_T, dim=1).to_dense().view(-1)
            _num_e = _tmp.size(0)
            self.cache["D_e"] = torch.sparse_csr_tensor(
                torch.arange(0, _num_e + 1),
                torch.arange(0, _num_e),
                _tmp,
                torch.Size([_num_e, _num_e]),
            )

        return self.cache["D_e"]

    def D_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e") is None:
            _tmp = (
                torch.sparse.sum(self._fetch_H().t(), dim=1).to_dense().clone().view(-1)
            )
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_e"]

    @property
    def D_e_neg_1(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e_neg_1") is None:
            _tmp = torch.sparse.sum(self.H_T, dim=1).to_dense().view(-1)
            _num_e = _tmp.size(0)
            _val = _tmp**-1
            _val[torch.isinf(_val)] = 0

            self.cache["D_e_neg_1"] = torch.sparse_csr_tensor(
                torch.arange(0, _num_e + 1),
                torch.arange(0, _num_e),
                _val,
                torch.Size([_num_e, _num_e]),
            )

        return self.cache["D_e_neg_1"]

    def D_e_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e_neg_1") is None:
            _mat = self.D_e_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_e_neg_1"]

    def N_e(self, v_idx: int) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Parameters:
            ``v_idx`` (``int``): The index of the vertex.
        """
        assert v_idx < self.num_v
        _tmp, e_bias = [], 0
        for name in self.group_names:
            _tmp.append(self.N_e_of_group(v_idx, name) + e_bias)
            e_bias += self.num_e_of_group(name)
        return torch.cat(_tmp, dim=0)

    def N_e_of_group(self, v_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Parameters:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        assert v_idx < self.num_v
        e_indices = self.H_of_group(group_name)[v_idx]._indices()[0]
        return e_indices.clone()

    def N_v(self, e_idx: int) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :attr:`num_e`).

        Parameters:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        assert e_idx < self.num_e
        for name in self.group_names:
            if e_idx < self.num_e_of_group(name):
                return self.N_v_of_group(e_idx, name)
            else:
                e_idx -= self.num_e_of_group(name)

    def N_v_of_group(self, e_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :func:`num_e_of_group`).

        Parameters:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        assert e_idx < self.num_e_of_group(group_name)
        v_indices = self.H_T_of_group(group_name)[e_idx]._indices()[0]
        return v_indices.clone()

    # =====================================================================================
    # spectral-based convolution/smoothing
    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        return super().smoothing(X, L, lamb)

    @property
    def L_sym(self) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_sym") is None:
            L_HGNN = self.L_HGNN.clone()
            self.cache["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack(
                    [
                        torch.arange(0, self.num_v).view(1, -1).repeat(2, 1),
                        L_HGNN.to_sparse_coo()._indices(),
                    ]
                ),
                torch.hstack(
                    [torch.ones(self.num_v), -L_HGNN.to_sparse_coo()._values()]
                ),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["L_sym"]

    def L_sym_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_sym") is None:
            L_HGNN = self.L_HGNN_of_group(group_name).clone()
            self.group_cache[group_name]["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack(
                    [
                        torch.arange(0, self.num_v).view(1, -1).repeat(2, 1),
                        L_HGNN._indices(),
                    ]
                ),
                torch.hstack([torch.ones(self.num_v), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["L_sym"]

    @property
    def L_rw(self) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top
        """
        if self.cache.get("L_rw") is None:
            _tmp = (
                self.D_v_neg_1.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T)
            )
            self.cache["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack(
                        [
                            torch.arange(0, self.num_v).view(1, -1).repeat(2, 1),
                            _tmp._indices(),
                        ]
                    ),
                    torch.hstack([torch.ones(self.num_v), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.cache["L_rw"]

    def L_rw_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_rw") is None:
            _tmp = (
                self.D_v_neg_1_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(
                    self.W_e_of_group(group_name),
                )
                .mm(
                    self.D_e_neg_1_of_group(group_name),
                )
                .mm(
                    self.H_T_of_group(group_name),
                )
            )
            self.group_cache[group_name]["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack(
                        [
                            torch.arange(0, self.num_v).view(1, -1).repeat(2, 1),
                            _tmp._indices(),
                        ]
                    ),
                    torch.hstack([torch.ones(self.num_v), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.group_cache[group_name]["L_rw"]

    ## HGNN Laplacian smoothing
    @property
    def L_HGNN(self) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_HGNN") is None:
            _d_v_neg_1_2 = self.D_v_neg_1_2.to_sparse_coo()
            _tmp = (
                _d_v_neg_1_2
                @ self.H
                @ self.W_e
                @ self.D_e_neg_1.to_sparse_coo()
                @ self.H_T
                @ _d_v_neg_1_2
            )
            self.cache["L_HGNN"] = _tmp.to_sparse_csr()
        return self.cache["L_HGNN"]

    def L_HGNN_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_HGNN") is None:
            _tmp = (
                self.D_v_neg_1_2_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name))
                .mm(
                    self.D_e_neg_1_of_group(group_name),
                )
                .mm(
                    self.H_T_of_group(group_name),
                )
                .mm(
                    self.D_v_neg_1_2_of_group(group_name),
                )
            )
            self.group_cache[group_name]["L_HGNN"] = _tmp.coalesce()
        return self.group_cache[group_name]["L_HGNN"]

    def smoothing_with_HGNN(
        self, X: torch.Tensor, drop_rate: float = 0.0
    ) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Parameters:
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        # if self.device != X.device:
        #     X = X.to(self.device)

        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN, drop_rate)
        else:
            L_HGNN = self.L_HGNN
        return L_HGNN.mm(X)

    def smoothing_with_HGNN_of_group(
        self, group_name: str, X: torch.Tensor, drop_rate: float = 0.0
    ) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Parameters:
            ``group_name`` (``str``): The name of the specified hyperedge group.
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN_of_group(group_name), drop_rate)
        else:
            L_HGNN = self.L_HGNN_of_group(group_name)
        return L_HGNN.mm(X)

    def smoothing_with_HWNN_approx(
        self,
        X: torch.Tensor,
        par: torch.nn.Parameter,
        W_d: torch.nn.Parameter,
        K1: int,
        K2: int,
        W: torch.nn.Parameter,
    ) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the approximated HWNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{theta}_{sum} \mathbf{Lambda}_{beta} \mathbf{theta'}_{sum}

        Parameters:
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``par`` (``torch.nn.Parameter``): A learnable parameter used in the HWNN approximation.
            ``W_d`` (``torch.nn.Parameter``): A trainable weight matrix for feature transformation.
            ``K1`` (``int``): The order of approximation for the first transformation step.
            ``K2`` (``int``): The order of approximation for the second transformation step.
            ``W`` (``torch.nn.Parameter``): A learnable weight matrix applied in the feature transformation step.
        """
        # if self.device != X.device:
        #     X = X.to(self.device)
        if self.device != W_d.device:
            W_d = W_d.to(self.device)
        if self.device != W.device:
            W = W.to(self.device)
        ncount = X.size()[0]
        W_d = torch.diag(W_d)
        Theta = self.L_HGNN
        Theta_t = torch.transpose(Theta, 0, 1)
        poly = par[0] * torch.eye(ncount).to(self.device)
        Theta_mul = torch.eye(ncount).to(self.device)
        for ind in range(1, K1):
            Theta_mul = Theta_mul @ Theta
            poly = poly + par[ind] * Theta_mul
        poly_t = par[K1] * torch.eye(ncount).to(self.device)
        Theta_mul = torch.eye(ncount).to(self.device)
        for ind in range(K1 + 1, K1 + K2):
            Theta_mul = Theta_mul @ Theta_t
            poly_t = poly_t + par[ind] * Theta_mul
        return poly @ W_d @ poly_t @ X @ W

    def smoothing_with_HWNN_wavelet(
        self, X: torch.Tensor, W_d: torch.nn.Parameter, W: torch.nn.Parameter
    ) -> torch.Tensor:
        r"""Return the smoothed feature matrix with original HWNN Laplacian matrix :


            .. math::
                \mathbf{X} = \mathbf{Psi}_{s} \mathbf{Lambda}_{beta} \mathbf{Psi}_{s}^{-1}

        Parameters:
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``par`` (``torch.nn.Parameter``): A learnable parameter used in the HWNN approximation.
            ``W_d`` (``torch.nn.Parameter``): A trainable weight matrix for feature transformation.
            ``K1`` (``int``): The order of approximation for the first transformation step.
            ``K2`` (``int``): The order of approximation for the second transformation step.
            ``W`` (``torch.nn.Parameter``): A learnable weight matrix applied in the feature transformation step.
        """
        # if self.device != X.device:
        #     X = X.to(self.device)
        if self.device != W_d.device:
            W_d = W_d.to(self.device)
        if self.device != W.device:
            W = W.to(self.device)
        W_d = torch.diag(W_d)
        Theta = self.L_HGNN
        Laplacian = torch.eye(Theta.size()[0]) - Theta
        fourier_e, fourier_v = torch.linalg.eigh(Laplacian, UPLO="U")
        wavelets = (
            fourier_v
            @ torch.diag(torch.exp(-1.0 * fourier_e))
            @ torch.transpose(fourier_v, 0, 1)
        )
        wavelets_inv = (
            fourier_v
            @ torch.diag(torch.exp(fourier_e))
            @ torch.transpose(fourier_v, 0, 1)
        )
        wavelets[wavelets < 0.00001] = 0
        wavelets_inv[wavelets_inv < 0.00001] = 0
        return wavelets @ W_d @ wavelets_inv @ X @ W

    # =====================================================================================
    # spatial-based convolution/message-passing
    # general message passing functions
    def v2e_aggregation(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``vertices to hyperedges``.

        Parameters:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyperedges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        # if self.device != X.device:
        #     self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T, drop_rate)
            else:
                P = self.H_T

            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight.shape[0]
            ), "The size of v2e_weight must be equal to the size of self.v2e_weight."
            P = torch.sparse_coo_tensor(
                self.H_T._indices(), v2e_weight, self.H_T.shape, device=self.device
            )

            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = torch.sparse.mm(D_e_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X

    def v2e_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``vertices to hyperedges`` in specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyperedges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        # if self.device != X.device:
        #     self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T_of_group(group_name), drop_rate)
            else:
                P = self.H_T_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight_of_group(group_name).shape[0]
            ), (
                "The size of v2e_weight must be equal to the size of"
                f" self.v2e_weight_of_group('{group_name}')."
            )
            P = torch.sparse_coo_tensor(
                self.H_T_of_group(group_name)._indices(),
                v2e_weight,
                self.H_T_of_group(group_name).shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X

    def v2e_update(self, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges``.

        Parameters:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        # if self.device != X.device:
        #     self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e, X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert (
                e_weight.shape[0] == self.num_e
            ), "The size of e_weight must be equal to the size of self.num_e."
            X = e_weight * X
        return X

    def v2e_update_of_group(
        self, group_name: str, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None
    ):
        r"""Message update step of ``vertices to hyperedges`` in specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        # if self.device != X.device:
        #     self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e_of_group(group_name), X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e_of_group(group_name), (
                "The size of e_weight must be equal to the size of"
                f" self.num_e_of_group('{group_name}')."
            )
            X = e_weight * X
        return X

    def v2e(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Parameters:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyperedges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """

        X = self.v2e_aggregation(X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update(X, e_weight)
        return X

    def v2e_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Parameters:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyperedges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        X = self.v2e_aggregation_of_group(
            group_name, X, aggr, v2e_weight, drop_rate=drop_rate
        )
        X = self.v2e_update_of_group(group_name, X, e_weight)
        return X

    def e2v_aggregation(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``hyperedges to vertices``.

        Parameters:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        # if self.device != X.device:
        #     self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H, drop_rate)
            else:
                P = self.H
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight.shape[0]
            ), "The size of e2v_weight must be equal to the size of self.e2v_weight."
            P = torch.sparse_coo_tensor(
                self.H._indices(),
                e2v_weight,
                self.H.shape,
                # device=self.device
            ).coalesce()

            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X

    def e2v_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``hyperedges to vertices`` in specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        # if self.device != X.device:
        #     self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_of_group(group_name), drop_rate)
            else:
                P = self.H_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1_of_group[group_name], X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight_of_group[group_name].shape[0]
            ), (
                "The size of e2v_weight must be equal to the size of"
                f" self.e2v_weight_of_group('{group_name}')."
            )
            P = torch.sparse_coo_tensor(
                self.H_of_group[group_name]._indices(),
                e2v_weight,
                self.H_of_group[group_name].shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X

    def e2v_update(self, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices``.

        Parameters:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        # if self.device != X.device:
        #     self.to(X.device)
        return X

    def e2v_update_of_group(self, group_name: str, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices`` in specified hyperedge group.

        Parameters:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        # if self.device != X.device:
        #     self.to(X.device)
        return X

    def e2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Parameters:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.e2v_aggregation(X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update(X)
        return X

    def e2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Parameters:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        X = self.e2v_aggregation_of_group(
            group_name, X, aggr, e2v_weight, drop_rate=drop_rate
        )
        X = self.e2v_update_of_group(group_name, X)
        return X

    def v2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices``. The combination of ``v2e`` and ``e2v``.

        Parameters:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e`` and ``e2v``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyperedges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e``. Default: ``None``.
        """
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate

        X = self.v2e(X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v(X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)

        return X

    def v2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices`` in specified hyperedge group. The combination of ``v2e_of_group`` and ``e2v_of_group``.

        Parameters:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e_of_group`` and ``e2v_of_group``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v_of_group``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyperedges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v_of_group``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e_of_group``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e_of_group``. Default: ``None``.
        """
        assert (
            group_name in self.group_names
        ), f"The specified {group_name} is not in existing hyperedge groups."
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e_of_group(
            group_name, X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate
        )
        X = self.e2v_of_group(
            group_name, X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate
        )
        return X

    def get_linegraph(self, s=1, weight=True):
        """
        Get the linegraph of the hypergraph based on the clique expansion.
        The edges will be the vertices of the line
        graph. Two vertices are connected by an s-line-graph edge if the
        corresponding hypergraph edges intersect in at least s hypergraph nodes.


        Parameters
        ----------
        s : Two vertices are connected if the nodes they correspond to share
        at least s incident hyper edges.
        edge : If edges=True (default)then the edges will be the vertices of the line
        graph. Two vertices are connected by an s-line-graph edge if the
        corresponding hypergraph edges intersect in at least s hypergraph nodes.
        If edges=False, the hypergraph nodes will be the vertices of the line
        graph.
        weight :

        Returns
        -------
            Graph: easygraph.Graph, the linegraph of the hypergraph.

        """
        edge_adjacency = self.edge_adjacency_matrix(s=s, weight=weight)
        graph = eg.from_scipy_sparse_matrix(edge_adjacency)
        return graph

    def get_clique_expansion(self, s=1, weight=True):
        """
        Get the linegraph of the hypergraph based on the clique expansion.
        The hypergraph nodes will be the vertices of the line
        graph. Two vertices are connected if the nodes they correspond to share
        at least s incident hyper edges.

        Parameters
        ----------
        s : Two vertices are connected if the nodes they correspond to share
        at least s incident hyper edges.
        edge : If edges=True (default)then the edges will be the vertices of the line
        graph. Two vertices are connected by an s-line-graph edge if the
        corresponding hypergraph edges intersect in at least s hypergraph nodes.
        If edges=False, the hypergraph nodes will be the vertices of the line
        graph.
        weight :

        Returns
        -------
            Graph: easygraph.Graph, the clique expansion of the hypergraph.

        """

        if self.cache.get("clique_expansion") is None:
            A = self.adjacency_matrix(s=s, weight=weight)
            graph = eg.Graph()
            A = np.array(np.nonzero(A))
            e1 = np.array([idx for idx in A[0]])
            e2 = np.array([idx for idx in A[1]])
            A = np.array([e1, e2]).T
            graph.add_edges_from(A)
            graph.add_nodes(list(range(0, self.num_v)))
            self.cache["clique_expansion"] = graph

        return self.cache["clique_expansion"]

    def cluster_coefficient(self):
        g = self.get_linegraph()
        return eg.clustering(g)

    def s_connected_components(self, s=1, edges=True, return_singletons=False):
        """
        Returns a generator for the :term:`s-edge-connected components
        <s-edge-connected component>`
        or the :term:`s-node-connected components <s-connected component,
        s-node-connected component>` of the hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        edges : boolean, optional, default = True
            If True will return edge components, if False will return node
            components
        return_singletons : bool, optional, default = False

        Notes
        -----
        If edges=True, this method returns the s-edge-connected components as
        lists of lists of edge uids.
        An s-edge-component has the property that for any two edges e1 and e2
        there is a sequence of edges starting with e1 and ending with e2
        such that pairwise adjacent edges in the sequence intersect in at least
        s nodes. If s=1 these are the path components of the hypergraph.

        If edges=False this method returns s-node-connected components.
        A list of sets of uids of the nodes which are s-walk connected.
        Two nodes v1 and v2 are s-walk-connected if there is a
        sequence of nodes starting with v1 and ending with v2 such that
        pairwise adjacent nodes in the sequence share s edges. If s=1 these
        are the path components of the hypergraph.

        Example
        -------
            >>> S = {'A':{1,2,3},'B':{2,3,4},'C':{5,6},'D':{6}}
            >>> H = Hypergraph(S)

            >>> list(H.s_components(edges=True))
            [{'C', 'D'}, {'A', 'B'}]
            >>> list(H.s_components(edges=False))
            [{1, 2, 3, 4}, {5, 6}]

        Yields
        ------
        s_connected_components : iterator
            Iterator returns sets of uids of the edges (or nodes) in the
            s-edge(node) components of hypergraph.

        """
        if not edges:
            g = self.get_clique_expansion()
        else:
            g = self.get_linegraph(s)
        for c in eg.connected_components(g):
            if not return_singletons and len(c) == 1:
                continue
            yield c

    @staticmethod
    def from_hypergraph_hypergcn(
        hypergraph,
        feature,
        with_mediator=False,
        remove_selfloop=True,
    ):
        r"""Construct a graph from a hypergraph with methods proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://arxiv.org/pdf/1809.02589.pdf>`_ paper .

        Args:
            ``hypergraph`` (``Hypergraph``): The source hypergraph.
            ``feature`` (``torch.Tensor``): The feature of the vertices.
            ``with_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
            ``remove_selfloop`` (``bool``): Whether to remove self-loop. Defaults to ``True``.
            ``device`` (``torch.device``): The device to store the graph. Defaults to ``torch.device("cpu")``.
        """

        num_v = hypergraph.num_v
        assert (
            num_v == feature.shape[0]
        ), "The number of vertices in hypergraph and feature.shape[0] must be equal!"
        e_list, new_e_list, new_e_weight = hypergraph.e[0], [], []
        rv = torch.rand((feature.shape[1], 1), device=feature.device)
        for e in e_list:
            num_v_in_e = len(e)
            # assert (
            #     num_v_in_e >= 2
            # ), "The number of vertices in an edge must be greater than or equal to 2!"
            p = torch.mm(feature[e, :], rv).squeeze()
            v_a_idx, v_b_idx = torch.argmax(p), torch.argmin(p)
            if not with_mediator:
                new_e_list.append((e[v_a_idx], e[v_b_idx]))
                new_e_weight.append(1.0 / num_v_in_e)
            else:
                w = 1.0 / (2 * num_v_in_e - 3)
                for mid_v_idx in range(num_v_in_e):
                    if mid_v_idx != v_a_idx and mid_v_idx != v_b_idx:
                        new_e_list.append([e[v_a_idx], e[mid_v_idx]])
                        new_e_weight.append(w)
                        new_e_list.append([e[v_b_idx], e[mid_v_idx]])
                        new_e_weight.append(w)
        # remove selfloop
        if remove_selfloop:
            new_e_list = torch.tensor(new_e_list, dtype=torch.long)
            new_e_weight = torch.tensor(new_e_weight, dtype=torch.float)
            e_mask = (new_e_list[:, 0] != new_e_list[:, 1]).bool()
            new_e_list = new_e_list[e_mask].numpy().tolist()
            new_e_weight = new_e_weight[e_mask].numpy().tolist()

        _g = eg.Graph()

        _g.add_nodes(list(range(0, num_v)))
        for (
            e,
            w,
        ) in zip(new_e_list, new_e_weight):
            if _g.has_edge(e[0], e[1]):
                _g.add_edge(e[0], e[1], weight=(w + _g.adj[e[0]][e[1]]["weight"]))
            else:
                _g.add_edge(e[0], e[1], weight=w)
        now_edges = []
        now_weight = []
        for e in _g.edges:
            now_edges.append((e[0], e[1]))
            now_weight.append(e[2]["weight"])
        now_edges.extend([(i, i) for i in range(num_v)])
        now_weight.extend([1.0] * num_v)
        _g.cache["e_both_side"] = (now_edges, now_weight)

        return _g
