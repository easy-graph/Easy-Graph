import os

import numpy as np
import scipy.sparse as sp

from easygraph.classes.graph import Graph

from .graph_dataset_base import EasyGraphBuiltinDataset
from .utils import _get_dgl_url
from .utils import _set_labels
from .utils import data_type_dict
from .utils import tensor


__all__ = [
    "AmazonCoBuyComputerDataset",
]


class GNNBenchmarkDataset(EasyGraphBuiltinDataset):
    r"""Base Class for GNN Benchmark dataset

    Reference: https://github.com/shchur/gnn-benchmark#datasets
    """

    def __init__(
        self, name, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        _url = _get_dgl_url("dataset/" + name + ".zip")
        super(GNNBenchmarkDataset, self).__init__(
            name=name,
            url=_url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        npz_path = os.path.join(self.raw_path, self.name + ".npz")
        g = self._load_npz(npz_path)
        # g = transforms.reorder_graph(
        #     g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
        self._graph = g
        self._data = [g]
        self._print_info()

    def has_cache(self):
        graph_path = os.path.join(self.save_path, "dgl_graph_v1.bin")
        if os.path.exists(graph_path):
            return True
        return False

    # def save(self):
    #     graph_path = os.path.join(self.save_path, 'dgl_graph_v1.bin')
    #     save_graphs(graph_path, self._graph)
    #
    # def load(self):
    #     graph_path = os.path.join(self.save_path, 'dgl_graph_v1.bin')
    #     graphs, _ = load_graphs(graph_path)
    #     self._graph = graphs[0]
    #     self._data = [graphs[0]]
    #     self._print_info()

    def _print_info(self):
        if self.verbose:
            print("  NumNodes: {}".format(self._graph.number_of_nodes()))
            print("  NumEdges: {}".format(2 * self._graph.number_of_edges()))
            print("  NumFeats: {}".format(self._graph.ndata["feat"].shape[-1]))
            print("  NumbClasses: {}".format(self.num_classes))

    def _load_npz(self, file_name):
        with np.load(file_name, allow_pickle=True) as loader:
            loader = dict(loader)
            num_nodes = loader["adj_shape"][0]
            adj_matrix = sp.csr_matrix(
                (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
                shape=loader["adj_shape"],
            ).tocoo()

            if "attr_data" in loader:
                # Attributes are stored as a sparse CSR matrix
                attr_matrix = sp.csr_matrix(
                    (
                        loader["attr_data"],
                        loader["attr_indices"],
                        loader["attr_indptr"],
                    ),
                    shape=loader["attr_shape"],
                ).todense()
            elif "attr_matrix" in loader:
                # Attributes are stored as a (dense) np.ndarray
                attr_matrix = loader["attr_matrix"]
            else:
                attr_matrix = None

            if "labels_data" in loader:
                # Labels are stored as a CSR matrix
                labels = sp.csr_matrix(
                    (
                        loader["labels_data"],
                        loader["labels_indices"],
                        loader["labels_indptr"],
                    ),
                    shape=loader["labels_shape"],
                ).todense()
            elif "labels" in loader:
                # Labels are stored as a numpy array
                labels = loader["labels"]
            else:
                labels = None
        if hasattr(adj_matrix, "format"):
            print("can be generate eg!")
        g = Graph(incoming_graph_data=adj_matrix)
        # g = transforms.to_bidirected(g)
        g = _set_labels(g, labels)
        g.ndata["feat"] = tensor(attr_matrix, data_type_dict()["float32"])
        g.ndata["label"] = tensor(labels, data_type_dict()["int64"])
        return g

    @property
    def num_classes(self):
        """Number of classes."""
        raise NotImplementedError

    def __getitem__(self, idx):
        r"""Get graph by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
        """
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._graph
        else:
            return self._transform(self._graph)

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return 1


class AmazonCoBuyComputerDataset(GNNBenchmarkDataset):
    r"""'Computer' part of the AmazonCoBuy dataset for node classification task.

    Amazon Computers and Amazon Photo are segments of the Amazon co-purchase graph [McAuley et al., 2015],
    where nodes represent goods, edges indicate that two goods are frequently bought together, node
    features are bag-of-words encoded product reviews, and class labels are given by the product category.

    Reference: `<https://github.com/shchur/gnn-benchmark#datasets>`_

    Statistics:

    - Nodes: 13,752
    - Edges: 491,722 (note that the original dataset has 245,778 edges but DGL adds
      the reverse edges and remove the duplicates, hence with a different number)
    - Number of classes: 10
    - Node feature size: 767

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    num_classes : int
        Number of classes for each node.

    Examples
    --------
    >>> data = AmazonCoBuyComputerDataset()
    >>> g = data[0]
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        super(AmazonCoBuyComputerDataset, self).__init__(
            name="amazon_co_buy_computer",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    @property
    def num_classes(self):
        """Number of classes.

        Return
        -------
        int
        """
        return 10
