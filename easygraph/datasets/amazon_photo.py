import os

import easygraph as eg
import numpy as np
import scipy.sparse as sp

from easygraph.classes.graph import Graph

from .graph_dataset_base import EasyGraphBuiltinDataset
from .utils import data_type_dict
from .utils import download
from .utils import extract_archive
from .utils import tensor


class AmazonPhotoDataset(EasyGraphBuiltinDataset):
    r"""Amazon Electronics Photo co-purchase graph dataset.

    Nodes represent products, and edges link products frequently co-purchased.
    Node features are bag-of-words of product reviews. The task is to classify
    the product category.

    Statistics:

    - Nodes: 7,650
    - Edges: 119,081
    - Number of Classes: 8
    - Features: 745

    Parameters
    ----------
    raw_dir : str, optional
        Raw file directory to download/contains the input data directory. Default: None
    force_reload : bool, optional
        Whether to reload the dataset. Default: False
    verbose : bool, optional
        Whether to print out progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~easygraph.Graph` object and returns
        a transformed version. The :class:`~easygraph.Graph` object will be
        transformed before every access.

    Examples
    --------
    >>> from easygraph.datasets import AmazonPhotoDataset
    >>> dataset = AmazonPhotoDataset()
    >>> g = dataset[0]
    >>> print(g.number_of_nodes())
    >>> print(g.number_of_edges())
    >>> print(g.nodes[0]['feat'].shape)
    >>> print(g.nodes[0]['label'])
    >>> print(dataset.num_classes)
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        name = "amazon_photo"
        url = "https://data.dgl.ai/dataset/amazon_co_buy_photo.zip"
        super(AmazonPhotoDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        path = os.path.join(self.raw_path, "amazon_co_buy_photo.npz")
        data = np.load(path)

        adj = sp.csr_matrix(
            (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
            shape=data["adj_shape"],
        )

        features = sp.csr_matrix(
            (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
            shape=data["attr_shape"],
        ).todense()

        labels = data["labels"]

        g = eg.Graph()
        g.add_edges_from(list(zip(*adj.nonzero())))

        for i in range(features.shape[0]):
            g.add_node(i, feat=np.array(features[i]).squeeze(), label=int(labels[i]))

        self._g = g
        self._num_classes = len(np.unique(labels))

        if self.verbose:
            print("Finished loading AmazonPhoto dataset.")
            print(f"  NumNodes: {g.number_of_nodes()}")
            print(f"  NumEdges: {g.number_of_edges()}")
            print(f"  NumFeats: {features.shape[1]}")
            print(f"  NumClasses: {self._num_classes}")

    def __getitem__(self, idx):
        assert idx == 0, "AmazonPhotoDataset only contains one graph"
        if self._g is None:
            raise ValueError("Graph has not been loaded or processed correctly.")
        return self._g if self._transform is None else self._transform(self._g)

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes
