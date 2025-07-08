"""CoauthorCS Dataset

This dataset contains a co-authorship network of authors who submitted papers to CS category.
Each node represents an author and edges represent co-authorships.
Node features are bag-of-words representations of keywords in the author's papers.
The task is node classification, with labels indicating the primary field of study.

Statistics:
- Nodes: 18333
- Edges: 81894
- Feature Dim: 6805
- Classes: 15

Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/cluster_gcn
"""

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


class CoauthorCSDataset(EasyGraphBuiltinDataset):
    r"""CoauthorCS citation network dataset.

    Nodes are authors, and edges indicate co-authorship relationships. Each node
    has a bag-of-words feature vector and a label denoting the primary research field.

    Parameters
    ----------
    raw_dir : str, optional
        Directory to store the raw downloaded files. Default: None
    force_reload : bool, optional
        Whether to re-download and process the dataset. Default: False
    verbose : bool, optional
        Whether to print detailed processing logs. Default: True
    transform : callable, optional
        Transform to apply to the graph on access.

    Examples
    --------
    >>> from easygraph.datasets import CoauthorCSDataset
    >>> dataset = CoauthorCSDataset()
    >>> g = dataset[0]
    >>> print("Nodes:", g.number_of_nodes())
    >>> print("Edges:", g.number_of_edges())
    >>> print("Feature shape:", g.nodes[0]['feat'].shape)
    >>> print("Label:", g.nodes[0]['label'])
    >>> print("Number of classes:", dataset.num_classes)
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        name = "coauthor_cs"
        url = "https://data.dgl.ai/dataset/coauthor_cs.zip"
        super(CoauthorCSDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        path = os.path.join(self.raw_path, "coauthor_cs.npz")
        data = np.load(path)

        # Reconstruct adjacency matrix
        adj = sp.csr_matrix(
            (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
            shape=data["adj_shape"],
        )

        # Reconstruct feature matrix
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
            print("Finished loading CoauthorCS dataset.")
            print(f"  NumNodes: {g.number_of_nodes()}")
            print(f"  NumEdges: {g.number_of_edges()}")
            print(f"  NumFeats: {features.shape[1]}")
            print(f"  NumClasses: {self._num_classes}")

    def __getitem__(self, idx):
        assert idx == 0, "CoauthorCSDataset only contains one graph"
        if self._g is None:
            raise ValueError("Graph has not been loaded or processed correctly.")
        return self._g if self._transform is None else self._transform(self._g)

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes
