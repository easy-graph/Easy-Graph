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


class RedditDataset(EasyGraphBuiltinDataset):
    r"""Reddit posts graph (Sept 2014) for community (subreddit) classification.

    Statistics:
    - Nodes: ~232,965
    - Edges: ~114 million (approx.)
    - Features per node: 602
    - Classes: number of subreddit communities

    Data are split by post-day: first 20 days train, then validation (30%), test (rest).

    Parameters
    ----------
    self_loop : bool
        Add self-loop edges if True.
    raw_dir, force_reload, verbose, transform : same as EasyGraphBuiltinDataset
    """

    def __init__(
        self,
        self_loop=False,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        transform=None,
    ):
        name = "reddit"
        url = "https://data.dgl.ai/dataset/reddit.zip"
        self.self_loop = self_loop
        super().__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        # Expect two files extracted: reddit_data.npz & reddit_graph.npz
        data = np.load(os.path.join(self.raw_path, "reddit_data.npz"))
        feat = data["feature"]  # shape [N, 602]
        labels = data["label"]  # shape [N]
        split = data["node_types"]  # 1=train,2=val,3=test

        # Load adjacency
        adj = sp.load_npz(os.path.join(self.raw_path, "reddit_graph.npz"))
        src, dst = adj.nonzero()
        if self.self_loop:
            self_loops = np.arange(adj.shape[0])
            src = np.concatenate([src, self_loops])
            dst = np.concatenate([dst, self_loops])
        edges = list(zip(src, dst))

        # Build graph
        g = eg.Graph()
        g.add_edges_from(edges)

        # Assign node features, labels, and masks
        for i in range(feat.shape[0]):
            g.add_node(
                i,
                feat=feat[i],
                label=int(labels[i]),
                train_mask=(split[i] == 1),
                val_mask=(split[i] == 2),
                test_mask=(split[i] == 3),
            )

        self._g = g
        self._num_classes = int(np.max(labels) + 1)

        if self.verbose:
            print("Loaded Reddit dataset:")
            print(f"  NumNodes: {g.number_of_nodes()}")
            print(f"  NumEdges: {g.number_of_edges()}")
            print(f"  NumFeats: {feat.shape[1]}")
            print(f"  NumClasses: {self._num_classes}")

    def __getitem__(self, idx):
        assert idx == 0, "RedditDataset only contains one graph"
        return self._g if self.transform is None else self.transform(self._g)

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes
