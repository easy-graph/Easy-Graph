import json
import os

import easygraph as eg
import numpy as np
import scipy.sparse as sp

from easygraph.classes.graph import Graph

from .graph_dataset_base import EasyGraphBuiltinDataset
from .utils import data_type_dict
from .utils import tensor


class FlickrDataset(EasyGraphBuiltinDataset):
    r"""Flickr dataset for node classification.

    Nodes are images and edges represent social tags co-occurrence.
    Node features are precomputed image embeddings. Labels indicate image categories.

    Statistics:
    - Nodes: 89,250
    - Edges: 899,756
    - Classes: 7
    - Feature dim: 500

    Source: GraphSAINT (https://arxiv.org/abs/1907.04931)

    Parameters
    ----------
    raw_dir : str, optional
        Custom directory to download the dataset. Default: None (uses standard cache dir).
    force_reload : bool, optional
        Whether to re-download and reprocess. Default: False.
    verbose : bool, optional
        Whether to print loading progress. Default: False.
    transform : callable, optional
        A transform applied to the graph on access.
    reorder : bool, optional
        Whether to apply graph reordering for locality (requires torch). Default: False.

    Examples
    --------
    >>> from easygraph.datasets import FlickrDataset
    >>> ds = FlickrDataset(verbose=True)
    >>> g = ds[0]
    >>> print(g.number_of_nodes(), g.number_of_edges(), ds.num_classes)
    >>> print(g.nodes[0]['feat'].shape, g.nodes[0]['label'])
    """

    def __init__(
        self,
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
        reorder=False,
    ):
        name = "flickr"
        url = self._get_dgl_url("dataset/flickr.zip")
        self._reorder = reorder
        super(FlickrDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        # Load adjacency
        coo = sp.load_npz(os.path.join(self.raw_path, "adj_full.npz"))
        g = eg.Graph()
        g.add_edges_from(list(zip(*coo.nonzero())))

        # Load features
        feats = np.load(os.path.join(self.raw_path, "feats.npy"))
        # Load labels
        with open(os.path.join(self.raw_path, "class_map.json")) as f:
            class_map = json.load(f)
            labels = np.array([class_map[str(i)] for i in range(feats.shape[0])])

        # Load train/val/test splits
        with open(os.path.join(self.raw_path, "role.json")) as f:
            role = json.load(f)
        train_mask = np.zeros(feats.shape[0], dtype=bool)
        train_mask[role["tr"]] = True
        val_mask = np.zeros(feats.shape[0], dtype=bool)
        val_mask[role["va"]] = True
        test_mask = np.zeros(feats.shape[0], dtype=bool)
        test_mask[role["te"]] = True

        # Attach node data
        for i in range(feats.shape[0]):
            g.add_node(i, feat=feats[i].astype(np.float32), label=int(labels[i]))
        g.graph["train_mask"] = train_mask
        g.graph["val_mask"] = val_mask
        g.graph["test_mask"] = test_mask

        self._g = g
        self._num_classes = int(labels.max() + 1)
        if self.verbose:
            print("Loaded Flickr dataset")
            print(
                f" Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}, Features: {feats.shape[1]}, Classes: {self._num_classes}"
            )

    def __getitem__(self, idx):
        assert idx == 0, "FlickrDataset contains only one graph"
        g = self._g
        # transfer mask info
        g.graph["train_mask"] = g.graph.pop("train_mask")
        g.graph["val_mask"] = g.graph.pop("val_mask")
        g.graph["test_mask"] = g.graph.pop("test_mask")
        return self._transform(g) if self._transform else g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def _get_dgl_url(path):
        from .utils import _get_dgl_url

        return _get_dgl_url(path)
